# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ..ernie.tokenizer import ErnieTokenizer

__all__ = ['ErnieMatchingTokenizer']


class ErnieMatchingTokenizer(ErnieTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-base-matching-query":
            "http://bj.bcebos.com/paddlenlp/models/transformers/ernie_base_matching/ernie-base-matching-query-vocab.txt",
            "ernie-base-matching-title":
            "http://bj.bcebos.com/paddlenlp/models/transformers/ernie_base_matching/ernie-base-matching-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-base-matching-query": {
            "do_lower_case": True
        },
        "ernie-base-matching-title": {
            "do_lower_case": True
        },
    }
