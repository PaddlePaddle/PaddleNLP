# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from functools import partial
import numpy as np

import paddle
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.transformers import BertModel, BertForSequenceClassification, BertTokenizer

from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.data import Stack, Tuple, Pad

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
    "chnsenticorp": Accuracy
}

MODEL_CLASSES = {"bert": (BertForSequenceClassification, BertTokenizer), }


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if 'text' in example:  # chnsenticorp
        example['sentence'] = example['text']
        example['labels'] = example['label']
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def create_glue_data_loader(task_name,
                            model_type="bert",
                            model_name_or_path="bert-base-uncased",
                            max_seq_length=128,
                            batch_size=64):
    task_name = task_name.lower()
    model_type = model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    metric_class = METRIC_CLASSES[task_name]
    if task_name == 'chnsenticorp':
        train_ds, dev_ds = load_dataset(task_name, splits=("train", "dev"))
    else:
        train_ds, dev_ds = load_dataset(
            'glue', task_name, splits=("train", "dev"))

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    if task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            'glue', task_name, splits=["dev_matched", "dev_mismatched"])
        dev_ds_matched = dev_dataset_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched, batch_size=batch_size, shuffle=False)
        dev_data_loader_matched = paddle.io.DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched, batch_size=batch_size, shuffle=False)
        dev_data_loader_mismatched = paddle.io.DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        return train_data_loader, dev_data_loader_matched, dev_data_loader_mismatched

    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=batch_size, shuffle=False)
    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    return train_data_loader, dev_data_loader
