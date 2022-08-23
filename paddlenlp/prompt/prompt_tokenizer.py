# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import itertools
import warnings
from functools import partial
from collections import defaultdict

import numpy as np

from .prompt_utils import InputFeatures

__all__ = ["MLMPromptTokenizer"]


class MLMPromptTokenizer(object):

    def __init__(self, tokenizer, max_seq_length, **kwargs):
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_length
        self._num_special_tokens = self._tokenizer.num_special_tokens_to_add()
        self._special_map = {
            "<cls>": "cls_token",
            "<sep>": "sep_token",
            "<pad>": "pad_token",
            "<unk>": "unk_token",
            "<mask>": "mask_token"
        }
        self.mask_token_id = self._tokenizer.mask_token_id
        self.pad_token_id = self._tokenizer.pad_token_id
        self.soft_token_id = self._tokenizer.unk_token_id

    def __call__(self, input_list):
        encoded_input = defaultdict(list)

        for input_dict in input_list:
            # Format text and special tokens, then convert them to ids.
            if input_dict["mask_ids"] == 1:
                text = [self.mask_token_id]

            if input_dict["text"] in self._special_map:
                special_token = getattr(self._tokenizer,
                                        self._special_map[input_dict["text"]])
                input_dict["text"] = special_token

            soft_ids = input_dict.get("soft_token_ids", None)
            if soft_ids is not None and soft_ids == 1:
                text = [self.soft_token_id]
            else:
                text = self._tokenizer.encode(
                    input_dict["text"],
                    add_special_tokens=False,
                    return_token_type_ids=False)["input_ids"]
            encoded_input["input_ids"].append(text)

            # Extend other features as the same length of input ids.
            for key in input_dict:
                if key != "text":
                    encoded_input[key].append([input_dict[key]] * len(text))

        max_seq_len = self._max_seq_len - self._num_special_tokens
        encoded_input = self.truncate(encoded_input, max_seq_len)
        encoded_input.pop("shortenable_ids")
        encoded_input = self.join(encoded_input)

        encoded_input = self.add_special_tokens(encoded_input)
        encoded_input = self.pad(encoded_input, self._max_seq_len,
                                 self.pad_token_id)
        return encoded_input

    def add_special_tokens(self, input_dict):
        for key in input_dict:
            new_inputs = self._tokenizer.build_inputs_with_special_tokens(
                input_dict[key])
            if key != "input_ids":
                special_mask = np.array(
                    self._tokenizer.get_special_tokens_mask(input_dict[key]))
                new_inputs = np.array(new_inputs)
                new_inputs[special_mask == 1] = 0
                new_inputs = new_inputs.tolist()
            input_dict[key] = new_inputs
        return input_dict

    @staticmethod
    def truncate(input_dict, max_seq_len):
        total_tokens = sum([len(text) for text in input_dict["input_ids"]])
        trunc_length = total_tokens - max_seq_len
        if trunc_length > 0:
            truncated_dict = defaultdict(list)
            trunc_mask = input_dict["shortenable_ids"]
            for key in input_dict:
                content = input_dict[key]
                count = trunc_length
                for idx, text in enumerate(content[::-1]):
                    index = -idx - 1
                    if len(text) == 0 or trunc_mask[index][0] == 0:
                        continue
                    if count < len(text):
                        content[index] = text[:-count]
                    else:
                        content[index] = []
                    count -= len(text)
                    if count <= 0:
                        break
                truncated_dict[key] = content
            return truncated_dict
        else:
            return input_dict

    @staticmethod
    def pad(input_dict, max_seq_len, pad_id, other_pad_id=0):
        for key, content in input_dict.items():
            if len(content) > max_seq_len:
                raise ValueError(
                    f"Truncated length of {key} is still longer than "
                    f"{max_seq_len}, please use a shorter prompt.")
            if key == "input_ids":
                pad_seq = [pad_id] * (max_seq_len - len(content))
            else:
                pad_seq = [other_pad_id] * (max_seq_len - len(content))
            input_dict[key].extend(pad_seq)
        return input_dict

    @staticmethod
    def join(input_dict):
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict
