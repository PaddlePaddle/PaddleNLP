# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ..bert.tokenizer import BertTokenizer

__all__ = ['MPNetTokenizer']


class MPNetTokenizer(BertTokenizer):
    """
    Construct a MPNet tokenizer. `MPNetTokenizer` is almost identical to `BertTokenizer` and runs end-to-end 
    tokenization: punctuation splitting and wordpiece. Refer to superclass `BertTokenizer` for usage examples 
    and documentation concerning parameters.
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "mpnet-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/mpnet/mpnet-base/vocab.txt",
        }
    }
    pretrained_init_configuration = {"mpnet-base": {"do_lower_case": True}}

    def __init__(self,
                 vocab_file,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="</s>",
                 pad_token="<pad>",
                 cls_token="<s>",
                 mask_token="<mask>"):

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token)
