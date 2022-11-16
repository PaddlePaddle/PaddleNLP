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
from typing import Any, Dict, List, Union

import numpy as np

__all__ = ["MLMPromptTokenizer"]


class MLMPromptTokenizer(object):

    omask_token = "[O-MASK]"

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, inputs: List[Dict[str, Any]]):
        part_text = [part["text"] for part in inputs]
        part_do_truncate = [part["do_truncate"] for part in inputs]
        max_lengths = self._create_max_lengths_from_do_truncate(
            part_text, part_do_truncate)

        encoded_inputs = defaultdict(list)
        option_length = None
        last_position = 1  # Id 0 denotes special token '[CLS]'.
        last_token_type = 0
        for index, part in enumerate(inputs):
            # Create input_ids.
            soft_token_ids = part.get("soft_tokens", None)
            if soft_token_ids is None or len(
                    soft_token_ids) == 1 and soft_token_ids[0] == 0:
                input_ids = self.tokenizer.encode(
                    part["text"],
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    truncation=True,
                    max_length=max_lengths[index])["input_ids"]
                encoded_inputs["soft_token_ids"].append([0] * len(input_ids))
            else:
                input_ids = soft_token_ids
                encoded_inputs["soft_token_ids"].append(soft_token_ids)
            encoded_inputs["input_ids"].append(input_ids)
            part_length = len(input_ids)

            # Create position_ids.
            position_ids, last_position = self._create_position_ids_from_part(
                input_ids, part, last_position)
            encoded_inputs["position_ids"].append(position_ids)

            # Create token_type_ids.
            if "token_types" in part:
                last_token_type = part["token_types"]
            encoded_inputs["token_type_ids"].append([last_token_type] *
                                                    part_length)

            # Create other features like encoder_ids.
            for name in part:
                if name not in [
                        "text", "soft_tokens", "positions", "token_types"
                ]:
                    encoded_inputs[name].append([part[name]] * part_length)

            # Record the length of options if exists.
            if self.omask_token in part["text"]:
                if option_length is not None:
                    raise ValueError(
                        "There are more than one sequence of options, which "
                        "will cause wrong attention masks.")
                option_length = len(input_ids)

        encoded_inputs.pop("do_truncate")
        encoded_inputs = self.join(encoded_inputs)
        encoded_inputs = self.add_special_tokens(encoded_inputs)
        attention_mask = self._create_attention_mask(
            encoded_inputs["input_ids"], option_length)
        if attention_mask is not None:
            encoded_inputs["attention_mask"] = attention_mask
        masked_positions = self._create_masked_positions(
            encoded_inputs["input_ids"], encoded_inputs["soft_token_ids"])
        if masked_positions is not None:
            encoded_inputs["masked_positions"] = masked_positions
        return encoded_inputs

    def _create_position_ids_from_part(self, input_ids: List[int],
                                       part: Dict[str,
                                                  Any], last_position: int):
        """ 
        Create position ids from prompt for each part.
        """
        part_length = len(input_ids)
        if "positions" in part and part["positions"] > 0:
            last_position = part["positions"]
        if self.omask_token in part["text"]:
            omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
            omask_index = [
                x for x in range(part_length) if input_ids[x] == omask_id
            ]
            omask_index = [0] + omask_index
            position_ids = []
            max_index = 0
            for start_id, end_id in zip(omask_index[:-1], omask_index[1:]):
                position_ids.extend(
                    list(range(last_position,
                               last_position + end_id - start_id)))
                max_index = max(end_id - start_id, max_index)
            if len(position_ids) < part_length:
                difference = part_length - len(position_ids)
                position_ids.extend(
                    range(last_position, last_position + difference))
                max_index = max(difference, max_index)
            last_position += max_index
        else:
            position_ids = list(
                range(last_position, last_position + part_length))
            last_position += part_length
        return position_ids, last_position

    def _create_max_lengths_from_do_truncate(self, part_text: List[str],
                                             part_do_truncate: List[bool]):
        """
        Create the max sequence length of each part.
        """
        text_length = sum([len(x) for x in part_text])
        if text_length < self.max_length:
            return [None] * len(part_text)

        num_special_token = self.tokenizer.num_special_tokens_to_add()
        cut_length = text_length - self.max_length + num_special_token
        max_lengths = []
        if self.tokenizer.truncation_side == "right":
            for index, part in enumerate(part_text[::-1]):
                if part_do_truncate[-1 - index] and cut_length > 0:
                    max_lengths.append(max(len(part) - cut_length, 0))
                    cut_length = cut_length - len(part)
                else:
                    max_lengths.append(None)
            max_lengths = max_lengths[::-1]
        else:
            for index, part in enumerate(text):
                if part_do_truncate[index] and cut_length > 0:
                    max_lengths.append(max(len(part) - cut_length, 0))
                    cut_length = cut_length - len(part)
                else:
                    max_lengths.append(None)
        return max_lengths

    def _create_attention_mask(self, input_ids: List[int],
                               option_length: Union[int, None]):
        if option_length is None:
            return None
        omask_id = self.tokenizer.convert_tokens_to_ids(self.omask_token)
        input_ids = np.array(input_ids)
        attention_mask = np.zeros([len(input_ids), len(input_ids)])
        pad_index = np.where(input_ids == self.tokenizer.pad_token_id)[0]
        attention_mask[:, pad_index] = 1
        attention_mask[pad_index, :] = 1
        omask_index = np.where(input_ids == omask_id)[0].tolist()
        opt_begin, opt_end = omask_index[0], omask_index[0] + option_length
        attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 1
        omask_index.append(opt_end)
        for opt_begin, opt_end in zip(omask_index[:-1], omask_index[1:]):
            attention_mask[opt_begin:opt_end, opt_begin:opt_end] = 0
        attention_mask = (1 - attention_mask) * -1e4
        return attention_mask

    def _create_masked_positions(self, input_ids: List[int],
                                 soft_token_ids: List[int]):
        non_soft_ids = np.array(input_ids) * (np.array(soft_token_ids) == 0)
        mask_id = self.tokenizer.mask_token_id

        masked_positions = np.where(non_soft_ids == mask_id)[0]
        if masked_positions.shape[0] == 0:
            return None
        return masked_positions.tolist()

    def add_special_tokens(self, input_dict: Dict[str, Any]):
        for key in input_dict:
            new_inputs = self.tokenizer.build_inputs_with_special_tokens(
                input_dict[key])
            if key != "input_ids":
                special_mask = np.array(
                    self.tokenizer.get_special_tokens_mask(input_dict[key]))
                new_inputs = np.array(new_inputs)
                # TODO (Huijuan): Use different ids according to specific keyword.
                new_inputs[special_mask == 1] = 0
                new_inputs = new_inputs.tolist()
            input_dict[key] = new_inputs
        return input_dict

    @staticmethod
    def join(input_dict):
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict
