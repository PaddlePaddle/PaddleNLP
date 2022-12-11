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

    def __call__(
        self,
        text,
        text_pair=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        padding=False,
        truncation=False,
        return_position_ids=False,
        return_token_type_ids=False,  # don't return token_type_ids
        return_attention_mask=False,
        return_length=False,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_dict=True,
        return_offsets_mapping=False,
        add_special_tokens=True,
        pad_to_multiple_of=None,
        return_tensors=None,
        verbose: bool = True,
        **kwargs
    ):
        return super().__call__(
            text,
            text_pair,
            max_length,
            stride,
            is_split_into_words,
            padding,
            truncation,
            return_position_ids,
            return_token_type_ids,
            return_attention_mask,
            return_length,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_dict,
            return_offsets_mapping,
            add_special_tokens,
            pad_to_multiple_of,
            return_tensors,
            verbose,
            **kwargs,
        )
