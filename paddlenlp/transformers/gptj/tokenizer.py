# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The Open AI Team Authors and The HuggingFace Inc. team
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

from .. import GPTTokenizer

__all__ = ["GPTJTokenizer"]


class GPTJTokenizer(GPTTokenizer):

    resource_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    pretrained_resource_files_map = {"vocab_file": {}, "merges_file": {}}
    pretrained_init_configuration = {}

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        max_len=None,
        pad_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        eol_token="\u010a",
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            max_len=max_len,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            eol_token=eol_token,
            **kwargs,
        )
