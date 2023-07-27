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


class InTokensMapDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.dataset = self._create_intokens_data()

    def _create_intokens_data(self):
        batch_records, max_len = [], 0
        cur_len_so_far = 0

        total_data = []
        for i in range(len(self.data)):
            record = self.data[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = _pad_batch_records(
                    batch_records,
                    pad_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    max_length=self.max_length,
                )
                total_data.append(padded_list)
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])

        # remaining data
        if batch_records:
            padded_list = _pad_batch_records(
                batch_records,
                pad_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                max_length=self.max_length,
            )
            total_data.append(padded_list)
        return total_data

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class InTokensIterableDataset(IterableDataset):
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
                padded_list = _pad_batch_records(
                    batch_records,
                    pad_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    max_length=self.max_length,
                )
                yield padded_list
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])

        if batch_records:
            padded_list = _pad_batch_records(
                batch_records,
                pad_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                max_length=self.max_length,
            )
            yield padded_list


def _pad_batch_records(batch_records, pad_id, bos_token_id, label_pad_id=-100, max_length=4096):

    keys = batch_records[0].keys()
    data_map = {}
    data_batch_map = {}
    for key in keys:
        if isinstance(batch_records[0][key], list):
            batch_record_token_ids = [record[key] for record in batch_records]
            # To adapt to chatglm position_2d
            if key == "position_ids":
                batch_token_ids = np.concatenate(batch_record_token_ids, axis=-1).tolist()
            else:
                batch_token_ids = sum(batch_record_token_ids, [])
            data_batch_map[key] = batch_record_token_ids
            # concated dataset
            data_map[key] = batch_token_ids

    batch_map = {}
    batch_map.update(data_map)
    batch_map["attention_mask"] = [np.array(record["attention_mask"]) for record in batch_records]
    batch_map["attention_mask"] = [np.tril(block_diag(*batch_map["attention_mask"])).tolist()]
    if "token_type_ids" in batch_map:
        batch_map.pop("token_type_ids")

    return batch_map
