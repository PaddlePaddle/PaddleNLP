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

import argparse
# import logging
import os
import sys
import random
import time
import math
import json
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import RobertaForSequenceClassification, RobertaTokenizer

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
    "roberta": (RobertaForSequenceClassification, RobertaTokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_type",
        default="ernie",
        type=str,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default="tmp",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.", )

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    args = parser.parse_args()
    return args


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['label']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)
    elif 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = tokenizer(
            sentence1, text_pair=example['abst'], max_seq_len=max_seq_length)
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                 )] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx + len(query)
                               )] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example = tokenizer(text, max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def do_test(args):
    paddle.set_device(args.device)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    train_ds, test_ds = load_dataset(
        'clue', args.task_name, splits=('train', 'test'))
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=False)
    test_data_loader = DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    model_class, _ = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.task_name == 'ocnli':
        args.task_name = 'ocnli_50k'
    f = open(
        os.path.join(args.output_dir, args.task_name + "_predict.json"), 'w')

    for step, batch in enumerate(test_data_loader):
        input_ids, segment_ids = batch

        with paddle.no_grad():
            logits = model(input_ids, segment_ids)

        preds = paddle.argmax(logits, axis=1)
        for idx, pred in enumerate(preds):
            j = json.dumps({"id": idx, "label": train_ds.label_list[pred]})
            f.write(j + "\n")


if __name__ == "__main__":
    args = parse_args()
    do_test(args)
