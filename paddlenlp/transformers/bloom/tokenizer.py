# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from ..tokenizer_utils import BPETokenizer


class BloomTokenizer(BPETokenizer):
    """
    Construct BloomTokenizer Based on BPETokenizer.

    Args:
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        unk_token (str, optional):
            The special token for unknown words.
            Defaults to "[UNK]".
        sep_token (str, optional):
            The special token for separator token.
            Defaults to "[SEP]".
        pad_token (str, optional):
            The special token for padding.
            Defaults to "[PAD]".
        cls_token (str, optional):
            The special token for cls.
            Defaults to "[CLS]".
        mask_token (str, optional):
            The special token for mask.
            Defaults to "[MASK]".
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        do_lower_case: bool = False,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        super(BloomTokenizer, self).__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
