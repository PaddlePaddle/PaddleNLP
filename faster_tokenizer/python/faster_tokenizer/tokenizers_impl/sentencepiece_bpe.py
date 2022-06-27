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

from .base_tokenizer import BaseFasterTokenizer
from faster_tokenizer.models import BPE
from faster_tokenizer.normalizers import NFKCNormalizer
from faster_tokenizer import Tokenizer
from faster_tokenizer.pretokenizers import MetaSpacePreTokenizer

__all__ = ['SentencePieceBPEFasterTokenizer']


class SentencePieceBPEFasterTokenizer(BaseFasterTokenizer):

    def __init__(self,
                 vocab=None,
                 merges=None,
                 unk_token="<unk>",
                 replacement="▁",
                 add_prefix_space=True,
                 dropout=None,
                 fuse_unk=False):
        if vocab is not None and merges is not None:
            tokenizer = Tokenizer(
                BPE(vocab,
                    merges,
                    dropout=dropout,
                    unk_token=unk_token,
                    fuse_unk=fuse_unk))
        else:
            tokenizer = Tokenizer(BPE())
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        tokenizer.normalizer = NFKCNormalizer()
        tokenizer.pretokenizer = MetaSpacePreTokenizer(
            replacement=replacement, add_prefix_space=add_prefix_space)
        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab_filename, merges_filename, **kwargs):
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)
        return SentencePieceBPEFasterTokenizer(vocab, merges, **kwargs)
