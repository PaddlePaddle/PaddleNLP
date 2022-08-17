# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.

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

__all__ = ['ErnieViL2Tokenizer']


class ErnieViL2Tokenizer(ErnieTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie_vit_b_16x":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_vil2/ernie_vit_b_16x/vocab.txt",
        }
    }
    pretrained_init_configuration = {"ernie_vit_b_16x": {"do_lower_case": True}}
