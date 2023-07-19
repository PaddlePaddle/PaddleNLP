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
from paddle.io import Dataset

from paddlenlp.trainer import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class CustomTrainer(Trainer):
    total_observed_tokens = 0.0

    def training_step(self, model, inputs):
        input_ids = inputs["input_ids"]
        self.total_observed_tokens += float(input_ids.shape[0] * input_ids.shape[1])
        return super().training_step(model, inputs)


class ProfilerCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, prof):
        self.prof = prof
        self.prof.start()

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()
        self.prof.summary()


class InTokensDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=4096, num_iter=100):
        self.tokenizer = tokenizer
        self.data = data
        self.records = data
        self.index = 0
        self.max_seq_len = max_seq_len
        self.num_iter = num_iter

    def __getitem__(self, idx):
        cur_len_so_far = 0
        # shuffle after each iteration
        if self.index >= len(self.records):
            np.random.shuffle(self.records)

        # keep index range 0~ len(dataset)
        self.index = self.index % len(self.records)
        cur_idx = self.index
        batch_records, max_len = [], 0
        # print(self.records[cur_idx:])
        for i in range(cur_idx, len(self.records)):
            # print(record)
            record = self.records[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_seq_len

            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            self.index += 1
        batch_list = _pad_batch_records(
            batch_records,
            pad_id=self.tokenizer.pad_token_id,
            start_id=self.tokenizer.bos_token_id,
            max_seq_len=self.max_seq_len,
        )
        return batch_list

    def __len__(self):
        return self.num_iter


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def _pad_batch_records(batch_records, pad_id, start_id, max_seq_len=4096):
    batch_record_token_ids = [record["input_ids"] for record in batch_records]  # leave one token for tgt_ids
    batch_token_ids = [sum(batch_record_token_ids, [])]

    # batch_position_ids = [record["position_ids"] for record in batch_records]
    # batch_position_ids = [sum(batch_position_ids, [])]

    # batch_position_ids_extra = [record["position_ids_extra"] for record in batch_records]
    # batch_position_ids_extra = [sum(batch_position_ids_extra, [])]

    # batch_loss_mask = [record["loss_mask"] for record in batch_records]
    # batch_loss_mask = [sum(batch_loss_mask, [])]

    batch_labels = [record["labels"] for record in batch_records]
    batch_labels = [sum(batch_labels, [])]

    padded_token_ids = pad_batch_data(
        batch_token_ids, pad_idx=pad_id, return_input_mask=False, max_seq_len=max_seq_len
    )
    # padded_position_ids = pad_batch_data(batch_position_ids, pad_idx=0, max_seq_len=self.max_seq_len)

    # padded_position_ids_extra = pad_batch_data(batch_position_ids_extra, pad_idx=0, max_seq_len=max_seq_len)
    # padded_batch_loss_mask = pad_batch_data(batch_loss_mask, pad_idx=0, max_seq_len=max_seq_len)
    padded_batch_labels = pad_batch_data(batch_labels, pad_idx=pad_id, max_seq_len=max_seq_len)
    # input_mask

    # add in-batch mask
    input_mask = _gen_self_attn_mask_for_glm_flatten(batch_record_token_ids, start_id, max_seq_len)
    return_list = {
        "input_ids": np.squeeze(padded_token_ids, axis=0).tolist(),
        #    "padded_position_ids_extra":np.squeeze(padded_position_ids_extra,axis=0),
        "attention_mask": np.squeeze(input_mask, axis=0).tolist(),
        "labels": np.squeeze(padded_batch_labels, axis=0).tolist(),
        #    "padded_batch_loss_mask":np.squeeze(padded_batch_loss_mask,axis=0),
    }
    return return_list


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
        if unbid_idx_1 != [] and unbid_idx_2 != []:  # mask the prompt for sentence embedding
            uns1_s, uns1_e = unbid_idx_1[index]
            uns2_s, uns2_e = unbid_idx_2[index]
            b[:, uns1_s:uns1_e] = 0
            b[:, uns2_s:uns2_e] = 0
            b[uns1_s:uns1_e, uns1_s:uns1_e] = 1
            b[uns1_s:uns1_e, uns2_s:uns2_e] = 1
            b[uns2_s:uns2_e, uns1_s:uns1_e] = 1
            b[uns2_s:uns2_e, uns2_s:uns2_e] = 1
        input_mask_data[0, 0, offset : offset + cur_len, offset : offset + cur_len] = b
        offset += cur_len

    return input_mask_data
