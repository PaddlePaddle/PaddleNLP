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

__all__ = ["ErnieViLTokenizer"]


class ErnieViLTokenizer(ErnieTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "ernie_vil-2.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_vil/ernie_vil-2.0-base-zh/vocab.txt",
            "disco_diffusion_ernie_vil-2.0-base-zh": "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_vil/disco_diffusion_ernie_vil-2.0-base-zh/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "ernie_vil-2.0-base-zh": {"do_lower_case": True},
        "disco_diffusion_ernie_vil-2.0-base-zh": {"do_lower_case": True},
    }
    max_model_input_sizes = {"ernie_vil-2.0-base-zh": 64, "disco_diffusion_ernie_vil-2.0-base-zh": 64}

    model_input_names = [
        "input_ids",
    ]
