# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ["ModernBertTokenizer"]


class ModernBertTokenizer(BertTokenizer):
    """
    Construct a ModernBERT tokenizer which is compatible with BERT tokenizer.
    """

    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "modernbert-base": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-base-vocab.txt",
            "modernbert-large": "https://bj.bcebos.com/paddlenlp/models/transformers/modernbert/modernbert-large-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "modernbert-base": {"do_lower_case": True},
        "modernbert-large": {"do_lower_case": True},
    }

    model_input_names = ["input_ids", "token_type_ids", "attention_mask", "position_ids"]
