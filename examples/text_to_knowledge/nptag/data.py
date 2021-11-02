# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json

import numpy as np
import paddle

def convert_example(example,
                    tokenzier,
                    max_seq_len=512,
                    max_cls_len=5,
                    summary_num=2):

    if len(example["text"]) + max_cls_len + 1 + summary_num + 1 > max_seq_len:
        example["text"] = example["text"][:(max_seq_len - (max_cls_len + 1 + summary_num + 1))]
    
    tokens = list(example["text"]) + ["是"] + ["[MASK]"] * max_cls_len
    inputs = tokenzier(
        tokens,
        is_split_into_words=True,
        pad_to_max_seq_len=True,
        max_seq_len=max_seq_len)

    label_indices = list(
        range(
            len(example["text"]) + 1 + summary_num,
            len(example["text"]) + 1 + max_cls_len + summary_num
        )
    )

    label_tokens = list(example["label"]) + ["[PAD]"] * (max_cls_len - len(example["label"]))
    labels = np.full([max_seq_len], fill_value=-100, dtype=np.int64)
    labels[label_indices] = tokenzier.convert_tokens_to_ids(label_tokens)

    return inputs["input_ids"], inputs["token_type_ids"], labels


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read_custom_data(filename):
    """Reads data."""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            yield {'text': text, 'label': label}


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab[line.strip()] = i
            i += 1
    return vocab