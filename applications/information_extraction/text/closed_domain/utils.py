# coding=utf-8
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

import json
import random

import numpy as np
import paddle

from paddlenlp.data.data_collator import DataCollatorForClosedDomainIE
from paddlenlp.utils.ie_utils import ClosedDomainIEProcessor


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def reader(data_path, tokenizer, max_seq_len=512, doc_stride=128, label_maps=None):
    with open(data_path, "r", encoding="utf-8") as f:
        examples = []
        for line in f:
            example = json.loads(line)
            examples.append(example)

    tokenized_examples = ClosedDomainIEProcessor.preprocess_text(
        examples,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        doc_stride=doc_stride,
        label_maps=label_maps,
    )
    for tokenized_example in tokenized_examples:
        yield tokenized_example


def get_eval_golds(data_path):
    golds = {"entity_list": [], "spo_list": []}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            golds["entity_list"].append(example["entity_list"])
            golds["spo_list"].append(example["spo_list"])
    if all([not spo for spo in golds["spo_list"]]):
        golds["spo_list"] = []
    return golds


def get_label_maps(label_maps_path=None):
    with open(label_maps_path, "r", encoding="utf-8") as fp:
        label_maps = json.load(fp)
    label_maps["entity_id2label"] = {val: key for key, val in label_maps["entity_label2id"].items()}
    label_maps["relation_id2label"] = {val: key for key, val in label_maps["relation_label2id"].items()}
    return label_maps


def create_dataloader(dataset, tokenizer=None, label_maps=None, batch_size=1, mode="train"):
    shuffle = True if mode == "train" else False
    batch_sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    data_collator = DataCollatorForClosedDomainIE(tokenizer, label_maps=label_maps)

    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, collate_fn=data_collator, num_workers=0, return_list=True
    )
    return dataloader
