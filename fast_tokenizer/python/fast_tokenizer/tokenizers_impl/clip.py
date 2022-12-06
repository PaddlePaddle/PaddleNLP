# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_tokenizer import BaseFastTokenizer

from fast_tokenizer.normalizers import NFCNormalizer, ReplaceNormalizer, LowercaseNormalizer, SequenceNormalizer
from fast_tokenizer.pretokenizers import SplitPreTokenizer, ByteLevelPreTokenizer, SequencePreTokenizer
from fast_tokenizer.models import BPE
from fast_tokenizer.postprocessors import RobertaPostProcessor
from fast_tokenizer import Tokenizer, SplitMode

__all__ = ["ClipFastTokenizer"]


class ClipFastTokenizer(BaseFastTokenizer):
    def __init__(
        self,
        vocab=None,
        merges=None,
        max_length=None,
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        continuing_subword_prefix="",
        end_of_word_suffix="</w>",
        trim_offsets=False,
    ):
        # Init Tokenizer instance using tokenization model
        tokenizer = Tokenizer(
            BPE(
                vocab,
                merges,
                unk_token=unk_token,
                continuing_subword_prefix=continuing_subword_prefix,
                end_of_word_suffix=end_of_word_suffix,
                fuse_unk=False,
            )
        )

        # Add special tokens
        bos_token_id = 0
        eos_token_id = 1
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(bos_token)) is not None:
            bos_token_id = tokenizer.token_to_id(str(bos_token))
            tokenizer.add_special_tokens([str(bos_token)])
        if tokenizer.token_to_id(str(eos_token)) is not None:
            eos_token_id = tokenizer.token_to_id(str(eos_token))
            tokenizer.add_special_tokens([str(eos_token)])

        # Set the normalizer
        tokenizer.normalizer = SequenceNormalizer(
            [NFCNormalizer(), ReplaceNormalizer(r"\s+", " "), LowercaseNormalizer()]
        )

        # Set the pretokenizer
        tokenizer.pretokenizer = SequencePreTokenizer(
            [
                SplitPreTokenizer(
                    r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                    split_mode=SplitMode.REMOVED,
                    invert=True,
                ),
                ByteLevelPreTokenizer(add_prefix_space=False),
            ]
        )

        # Set the postprocessor
        tokenizer.postprocessor = RobertaPostProcessor(
            sep=(eos_token, eos_token_id), cls=(bos_token, bos_token_id), trim_offsets=False, add_prefix_space=False
        )

        parameters = {
            "model": "BPE",
            "unk_token": unk_token,
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "add_prefix_space": add_prefix_space,
            "max_length": max_length,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "trim_offsets": trim_offsets,
        }
        super().__init__(tokenizer, parameters)
