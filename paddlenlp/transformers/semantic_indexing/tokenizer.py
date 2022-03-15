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

__all__ = ['ErnieSemanticIndexingTokenizer']


class ErnieSemanticIndexingTokenizer(ErnieTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie-base-cn-query":
            "http://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie-base-cn-query-vocab.txt",
            "ernie-base-cn-title":
            "http://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie-base-cn-title-vocab.txt",
            "ernie-base-en-query":
            "http://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie-base-en-query-vocab.txt",
            "ernie-base-en-title":
            "http://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie-base-en-title-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie-base-cn-query": {
            "do_lower_case": True
        },
        "ernie-base-cn-title": {
            "do_lower_case": True
        },
        "ernie-base-en-query": {
            "do_lower_case": True
        },
        "ernie-base-en-title": {
            "do_lower_case": True
        },
    }
