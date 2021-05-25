# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['TinyBertTokenizer']


class TinyBertTokenizer(BertTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "tinybert-4l-312d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-vocab.txt",
            "tinybert-6l-768d":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-vocab.txt",
            "tinybert-4l-312d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-v2-vocab.txt",
            "tinybert-6l-768d-v2":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-v2-vocab.txt",
            "tinybert-4l-312d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-4l-312d-zh-vocab.txt",
            "tinybert-6l-768d-zh":
            "http://paddlenlp.bj.bcebos.com/models/transformers/tinybert/tinybert-6l-768d-zh-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "tinybert-4l-312d": {
            "do_lower_case": True
        },
        "tinybert-6l-768d": {
            "do_lower_case": True
        },
        "tinybert-4l-312d-v2": {
            "do_lower_case": True
        },
        "tinybert-6l-768d-v2": {
            "do_lower_case": True
        },
        "tinybert-4l-312d-zh": {
            "do_lower_case": True
        },
        "tinybert-6l-768d-zh": {
            "do_lower_case": True
        },
    }
