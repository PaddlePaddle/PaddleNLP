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

__all__ = ['VisualBertTokenizer']


class VisualBertTokenizer(BertTokenizer):
    """
    Constructs a VisualBert tokenizer. `VisualBertTokenizer` is identical to `BertTokenizer` and runs end-to-end 
    tokenization: punctuation splitting and wordpiece. Refer to superclass `BertTokenizer` for usage examples 
    and documentation concerning parameters.
    """
