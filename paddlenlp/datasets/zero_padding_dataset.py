# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from paddle.io import Dataset, IterableDataset
from scipy.linalg import block_diag


def generate_greedy_packs(examples, max_length):
    left_len = np.zeros([len(examples)]) - 1
    left_len[0] = max_length  # At the beginning, only the first pack is valid.
    generate_packs = [[] for i in range(len(examples))]
    index, left_index = 0, 0

    while index < len(examples):
        record = examples[index]
        max_left_index = left_len.argmax()
        # Put the current sequence into the largest left space valid pack.
        if len(record["input_ids"]) <= left_len[max_left_index]:
            generate_packs[max_left_index].append(record)
            left_len[max_left_index] -= len(record["input_ids"])
            index += 1
        else:
            left_index += 1
            left_len[left_index] = max_length

    return generate_packs


class ZeroPadding:
    required_output_keys = ["input_ids", "labels", "attention_mask"]
    # Only supported the following keys for ZeroPadding. Keys outside of the set will be ignored.
    supported_input_keys = [
        "input_ids",
        "labels",
        "attention_mask",
        "position_ids",
        "chosen_labels",
        "rejected_labels",
        "response_indexs",
        "attn_mask_startend_row_indices",
    ]

    @classmethod
    def _pad_batch_records_to_max_length(cls, batch_records, max_length, pad_token=0):
        # confirm the at least one item in the pack
        if len(batch_records) == 0:
            return batch_records
        # count all records total length
        total_length = sum([len(record["input_ids"]) for record in batch_records])
        reserved_length = max_length - total_length

        # append padding to the max_length
        if "attn_mask_startend_row_indices" in batch_records[0]:
            # attn_mask_startend_row_indices is a list of row indices `0`,
            # which indicates that all tokens are masked.
            batch_records.append(
                {
                    "input_ids": [pad_token] * reserved_length,
                    "labels": [-100] * reserved_length,
                    "attn_mask_startend_row_indices": [0] * reserved_length,
                }
            )
        elif "attention_mask" in batch_records[0]:
            # attention_mask is a fullly masked attention matrix (all False)
            # which indicates that all tokens are masked.
            batch_records.append(
                {
                    "input_ids": [pad_token] * reserved_length,
                    "labels": [-100] * reserved_length,
                    "attention_mask": np.zeros((reserved_length, reserved_length), dtype=bool),
                }
            )

        return batch_records

    @classmethod
    def _pad_batch_records(cls, batch_records, max_length):
        batch_records = cls._pad_batch_records_to_max_length(batch_records, max_length)

        # Only consider supported input keys
        input_keys = [key for key in batch_records[0].keys() if key in cls.supported_input_keys]
        if "attn_mask_startend_row_indices" not in input_keys and "attention_mask" not in input_keys:
            input_keys.append("attention_mask")
        batched_features = {key: [] for key in input_keys}
        sequence_sum = 0
        for record in batch_records:
            batched_features["input_ids"].extend(record["input_ids"])
            if "labels" in record:
                batched_features["labels"].extend(record["labels"])
            elif "rejected_labels" in input_keys and "chosen_labels" in input_keys:
                batched_features["rejected_labels"].extend(record["rejected_labels"])
                batched_features["chosen_labels"].extend(record["chosen_labels"])
                response_indexs = [
                    record["response_indexs"][0] + sequence_sum,  # chosen_response_start_index
                    record["response_indexs"][1] + sequence_sum,  # rejeted_response_start_index
                    record["response_indexs"][2] + sequence_sum,  # rejeted_response_end_index + 1
                ]
                batched_features["response_indexs"].append(response_indexs)
            else:
                raise ValueError("labels is required for ZeroPadding Dataset")

            seq_length = len(record["input_ids"])
            # If attention_mask is not given, assume it's causal mask
            if "attn_mask_startend_row_indices" in record:
                attn_mask_startend_row_indices = [i + sequence_sum for i in record["attn_mask_startend_row_indices"]]
                batched_features["attn_mask_startend_row_indices"].extend(attn_mask_startend_row_indices)
            else:
                attention_mask = record.get("attention_mask", np.tril(np.ones([seq_length, seq_length], dtype=bool)))
                batched_features["attention_mask"].append(attention_mask)
            # NOTE: position_ids is optional and not required by every model
            # We append instead of extend here to accomodate 2D position ids
            if "position_ids" in record:
                batched_features["position_ids"].append(record["position_ids"])
            sequence_sum += seq_length

        if "attention_mask" in batched_features:
            block_attention_mask = block_diag(*batched_features["attention_mask"])
            # convert to 3-D [batch_size(1), seq_length, seq_length]
            batched_features["attention_mask"] = np.expand_dims(block_attention_mask, axis=0)
        if "position_ids" in batched_features:
            # Accomodate both 1D and 2D position ids
            batched_features["position_ids"] = np.concatenate(batched_features["position_ids"], axis=-1).tolist()
        return batched_features


class ZeroPaddingMapDataset(ZeroPadding, Dataset):
    def __init__(self, data, tokenizer, max_length, greedy_zero_padding=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.greedy_zero_padding = greedy_zero_padding
        self.new_data = self._create_zero_padding_data(data)

    def _create_zero_padding_data(self, data):
        total_data = []
        if not self.greedy_zero_padding:
            batch_records = []
            cur_len_so_far = 0
            for i in range(len(data)):
                record = data[i]
                if len(record["input_ids"]) > self.max_length:
                    continue
                to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
                if to_append:
                    batch_records.append(record)
                    cur_len_so_far += len(record["input_ids"])
                else:
                    # exceed max length
                    padded_list = self._pad_batch_records(batch_records, self.max_length)
                    total_data.append(padded_list)
                    # reset
                    batch_records = []
                    cur_len_so_far = 0
                    # append current data
                    batch_records.append(record)
                    cur_len_so_far += len(record["input_ids"])

            # remaining data
            if batch_records:
                padded_list = self._pad_batch_records(batch_records, self.max_length)
                total_data.append(padded_list)
        else:
            examples = []
            buffer_size = 500
            i = 0
            for record in data:
                if len(record["input_ids"]) > self.max_length:
                    continue
                if i < buffer_size:
                    examples.append(record)
                    i += 1
                else:
                    # Running greedy strategy in examples.
                    generate_packs = generate_greedy_packs(examples, self.max_length)
                    for batch_records in generate_packs:
                        if len(batch_records) > 0:
                            padded_list = self._pad_batch_records(batch_records, self.max_length)
                            total_data.append(padded_list)
                    examples = [record]
                    i = 1
            if len(examples) > 0:
                generate_packs = generate_greedy_packs(examples, self.max_length)
                for batch_records in generate_packs:
                    if len(batch_records) > 0:
                        padded_list = self._pad_batch_records(batch_records, self.max_length)
                        total_data.append(padded_list)

        return total_data

    def __getitem__(self, idx):
        return self.new_data[idx]

    def __len__(self):
        return len(self.new_data)


class ZeroPaddingIterableDataset(ZeroPadding, IterableDataset):
    def __init__(self, data, tokenizer, max_length, greedy_zero_padding=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.zero_padding_global_step = 0
        self.greedy_zero_padding = greedy_zero_padding

    def __iter__(self):
        if not self.greedy_zero_padding:
            batch_records = []
            cur_len_so_far = 0
            for record in self.data:
                to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
                if to_append:
                    batch_records.append(record)
                    self.zero_padding_global_step += 1
                    cur_len_so_far += len(record["input_ids"])
                else:
                    # exceed max length
                    padded_list = self._pad_batch_records(batch_records, self.max_length)
                    yield padded_list
                    # reset
                    batch_records = []
                    cur_len_so_far = 0
                    # append current data
                    batch_records.append(record)
                    self.zero_padding_global_step += 1
                    cur_len_so_far += len(record["input_ids"])
            if batch_records:
                padded_list = self._pad_batch_records(batch_records, self.max_length)
                yield padded_list
        else:
            examples = []
            buffer_size = 500
            i = 0
            for record in self.data:
                if len(record["input_ids"]) > self.max_length:
                    continue
                if i < buffer_size:
                    examples.append(record)
                    self.zero_padding_global_step += 1
                    i += 1
                else:
                    # Running greedy strategy in examples.
                    generate_packs = generate_greedy_packs(examples, self.max_length)
                    for batch_records in generate_packs:
                        if len(batch_records) > 0:
                            padded_list = self._pad_batch_records(batch_records, self.max_length)
                            yield padded_list
                    examples = [record]
                    self.zero_padding_global_step += 1
                    i = 1
            if len(examples) > 0:
                generate_packs = generate_greedy_packs(examples, self.max_length)
                for batch_records in generate_packs:
                    if len(batch_records) > 0:
                        padded_list = self._pad_batch_records(batch_records, self.max_length)
                        yield padded_list
