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


class InTokensMapDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=4096):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len
        self.dataset = self._intokens()

    def _intokens(
        self,
    ):
        batch_records, max_len = [], 0
        cur_len_so_far = 0

        total_data = []
        for i in range(len(self.data)):
            record = self.data[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_seq_len
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = _pad_batch_records(
                    batch_records,
                    pad_id=self.tokenizer.pad_token_id,
                    start_id=self.tokenizer.bos_token_id,
                    max_seq_len=self.max_seq_len,
                )
                total_data.append(padded_list)
                batch_records, max_len = [], 0
                cur_len_so_far = 0
        # remaining data
        if batch_records:
            padded_list = _pad_batch_records(
                batch_records,
                pad_id=self.tokenizer.pad_token_id,
                start_id=self.tokenizer.bos_token_id,
                max_seq_len=self.max_seq_len,
            )
            total_data.append(padded_list)
        return total_data

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class InTokensIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, max_seq_len=4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        for i in range(len(self.data)):
            record = self.data[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_seq_len
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = _pad_batch_records(
                    batch_records,
                    pad_id=self.tokenizer.pad_token_id,
                    start_id=self.tokenizer.bos_token_id,
                    max_seq_len=self.max_seq_len,
                )
                yield padded_list
                batch_records, max_len = [], 0
                cur_len_so_far = 0

        if batch_records:
            padded_list = _pad_batch_records(
                batch_records,
                pad_id=self.tokenizer.pad_token_id,
                start_id=self.tokenizer.bos_token_id,
                max_seq_len=self.max_seq_len,
            )
            yield padded_list


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    # [1,max_seq_lens]
    inst_data = [inst + list([pad_idx] * (max_len - len(inst))) for inst in insts]
    # [max_seq_lens]
    return inst_data[0]


def _pad_batch_records(batch_records, pad_id, start_id, label_pad_id=-100, max_seq_len=4096):

    keys = batch_records[0].keys()
    data_map = {}
    data_batch_map = {}
    for key in keys:
        if isinstance(batch_records[0][key], list):
            batch_record_token_ids = [record[key] for record in batch_records]
            batch_token_ids = [sum(batch_record_token_ids, [])]
            data_batch_map[key] = batch_record_token_ids
            data_map[key] = batch_token_ids

    batch_map = {}
    batch_map["input_ids"] = pad_batch_data(data_map["input_ids"], pad_idx=pad_id, max_seq_len=max_seq_len)
    batch_map["position_ids"] = pad_batch_data(data_map["position_ids"], pad_idx=0, max_seq_len=max_seq_len)
    batch_map["labels"] = pad_batch_data(data_map["labels"], pad_idx=label_pad_id, max_seq_len=max_seq_len)
    # input_mask

    # add in-batch mask
    batch_map["attention_mask"] = _gen_self_attn_mask_for_glm_flatten(
        data_batch_map["input_ids"], start_id, max_seq_len
    )
    batch_map["attention_mask"] = np.squeeze(batch_map["attention_mask"], axis=0).tolist()
    return batch_map


def _gen_self_attn_mask_for_glm_flatten(
    batch_token_ids, start_id, batch_size_fact=None, unbid_idx_1=[], unbid_idx_2=[]
):
    assert (
        len(sum(batch_token_ids, [])) <= batch_size_fact
    ), f"{len(sum(batch_token_ids, []))} > {batch_size_fact} is not allowed"
    # Note(gongenlei): unsqueeze attention mask to 4 dims
    input_mask_data = np.zeros((1, 1, batch_size_fact, batch_size_fact), dtype="float32")
    offset = 0
    for index, token_ids in enumerate(batch_token_ids):
        cur_len = len(token_ids)

        b = np.tril(np.ones([cur_len, cur_len]), 0)
        if start_id not in batch_token_ids[index]:
            first_start_index = 0
        else:
            first_start_index = batch_token_ids[index].index(start_id)
        b[:first_start_index, :first_start_index] = 1  # bi-directional attention before the first [START]
        input_mask_data[0, 0, offset : offset + cur_len, offset : offset + cur_len] = b
        offset += cur_len

    return input_mask_data
