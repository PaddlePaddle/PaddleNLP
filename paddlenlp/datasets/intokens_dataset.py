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


class InTokens:
    required_input_keys = ["input_ids", "labels"]
    required_output_keys = ["input_ids", "labels", "attention_mask"]
    # Only supported the following keys for InTokens. Keys outside of the set will be ignored.
    supported_input_keys = ["input_ids", "labels", "attention_mask", "position_ids"]

    @classmethod
    def _pad_batch_records(cls, batch_records):
        # TODO: support pad_to_max_length for Pipeline parallel
        # Only consider supported input keys
        input_keys = [key for key in batch_records[0].keys() if key in cls.supported_input_keys]

        # Check required_keys
        for key in cls.required_input_keys:
            if key not in input_keys:
                raise ValueError(f"feature `{key}` is required for InTokensDataset")
        # Output features must include all required output keys
        for key in cls.required_output_keys:
            if key not in input_keys:
                input_keys.append(key)

        batched_features = {key: [] for key in input_keys}
        for record in batch_records:
            batched_features["input_ids"].extend(record["input_ids"])
            batched_features["labels"].extend(record["labels"])
            seq_length = len(record["input_ids"])
            # If attention_mask is not given, assume it's causal mask
            attention_mask = record.get("attention_mask", np.tril(np.ones([seq_length, seq_length], dtype=bool)))
            batched_features["attention_mask"].append(attention_mask)
            # NOTE: position_ids is optional and not required by every model
            # We append instead of extend here to accomodate 2D position ids
            if "position_ids" in record:
                batched_features["position_ids"].append(record["position_ids"])
        block_attention_mask = block_diag(*batched_features["attention_mask"])
        # convert to 3-D [batch_size(1), seq_length, seq_length]
        batched_features["attention_mask"] = np.expand_dims(block_attention_mask, axis=0)
        if "position_ids" in batched_features:
            # Accomodate both 1D and 2D position ids
            batched_features["position_ids"] = np.concatenate(batched_features["position_ids"], axis=-1).tolist()
        return batched_features


class InTokensMapDataset(InTokens, Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.new_data = self._create_intokens_data(data)

    def _create_intokens_data(self, data):
        batch_records, max_len = [], 0
        cur_len_so_far = 0

        total_data = []
        for i in range(len(data)):
            record = data[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = self._pad_batch_records(batch_records)
                total_data.append(padded_list)
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])

        # remaining data
        if batch_records:
            padded_list = self._pad_batch_records(batch_records)
            total_data.append(padded_list)
        return total_data

    def __getitem__(self, idx):
        return self.new_data[idx]

    def __len__(self):
        return len(self.new_data)


class InTokensIterableDataset(InTokens, IterableDataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        for record in self.data:
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = self._pad_batch_records(batch_records)
                yield padded_list
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
        if batch_records:
            padded_list = self._pad_batch_records(batch_records)
            yield padded_list
