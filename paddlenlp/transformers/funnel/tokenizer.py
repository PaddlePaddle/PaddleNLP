# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

__all__ = ["FunnelTokenizer"]

import os
from typing import List, Optional

from .. import BasicTokenizer, WordpieceTokenizer
from ..bert.tokenizer import BertTokenizer


class FunnelTokenizer(BertTokenizer):
    cls_token_type_id = 2
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "funnel-transformer/small": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small/vocab.txt",
            "funnel-transformer/small-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small-base/vocab.txt",
            "funnel-transformer/medium": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium/vocab.txt",
            "funnel-transformer/medium-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium-base/vocab.txt",
            "funnel-transformer/intermediate": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate/vocab.txt",
            "funnel-transformer/intermediate-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate-base/vocab.txt",
            "funnel-transformer/large": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large/vocab.txt",
            "funnel-transformer/large-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large-base/vocab.txt",
            "funnel-transformer/xlarge": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge/vocab.txt",
            "funnel-transformer/xlarge-base": "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge-base/vocab.txt",
        },
    }
    pretrained_init_configuration = {
        "funnel-transformer/small": {"do_lower_case": True},
        "funnel-transformer/small-base": {"do_lower_case": True},
        "funnel-transformer/medium": {"do_lower_case": True},
        "funnel-transformer/medium-base": {"do_lower_case": True},
        "funnel-transformer/intermediate": {"do_lower_case": True},
        "funnel-transformer/intermediate-base": {"do_lower_case": True},
        "funnel-transformer/large": {"do_lower_case": True},
        "funnel-transformer/large-base": {"do_lower_case": True},
        "funnel-transformer/xlarge": {"do_lower_case": True},
        "funnel-transformer/xlarge-base": {"do_lower_case": True},
    }

    max_model_input_sizes = {
        "funnel-transformer/small": 512,
        "funnel-transformer/small-base": 512,
        "funnel-transformer/medium": 512,
        "funnel-transformer/medium-base": 512,
        "funnel-transformer/intermediate": 512,
        "funnel-transformer/intermediate-base": 512,
        "funnel-transformer/large": 512,
        "funnel-transformer/large-base": 512,
        "funnel-transformer/xlarge": 512,
        "funnel-transformer/xlarge-base": 512,
    }

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        do_basic_tokenize=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = FunnelTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=unk_token)

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Funnel
        Transformer sequence pair mask has the following format:
        ```
        2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
