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

import paddle
from paddlenlp.datasets import MapDataset


def load_dict(dict_path):
    vocab = {}
    i = 0
    with open(dict_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            vocab[line.strip()] = i
            i += 1
    return vocab


def convert_example(example,
                    tokenizer,
                    max_seq_len,
                    tags_to_idx=None,
                    summary_num=2,
                    is_test=False):
    tokens = example["tokens"]
    tokenized_input = tokenizer(tokens,
                                return_length=True,
                                is_split_into_words=True,
                                max_seq_len=max_seq_len)

    if is_test:
        return tokenized_input['input_ids'], tokenized_input[
            'token_type_ids'], tokenized_input['seq_len']

    tags = example["tags"]
    if len(tokenized_input['input_ids']) - 1 - summary_num < len(tags):
        tags = tags[:len(tokenized_input['input_ids']) - 1 - summary_num]
    # '[CLS]' and '[SEP]' will get label 'O'
    tags = ['O'] * (summary_num) + tags + ['O']
    tags += ['O'] * (len(tokenized_input['input_ids']) - len(tags))
    tokenized_input['tags'] = [tags_to_idx[x] for x in tags]
    return tokenized_input['input_ids'], tokenized_input[
        'token_type_ids'], tokenized_input['seq_len'], tokenized_input['tags']


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset,
                                batch_sampler=batch_sampler,
                                collate_fn=batchify_fn,
                                return_list=True)


def read_custom_data(filename):
    """Reads data"""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            example = transfer_str_to_example(line.strip())
            yield example


def transfer_str_to_example(sample):
    text = ""
    tags = []
    items = sample.split(" ")
    items = [item.rsplit("/", 1) for item in items]
    for w, t in items:
        text += w
        if len(w) == 1:
            tags.append(f"S-{t}")
        else:
            l = len(w)
            for j in range(l):
                if j == 0:
                    tags.append(f"B-{t}")
                elif j == l - 1:
                    tags.append(f"E-{t}")
                else:
                    tags.append(f"I-{t}")
    res = {
        "tokens": list(text),
        "tags": tags,
    }
    return res
