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

import argparse
import functools
import os
import random

import numpy as np
import paddle
from paddle.io import BatchSampler, DataLoader
from trustai.interpretation import RepresenterPointModel

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True, type=str, help="The dataset directory.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument("--dirty_num", type=int, default=100, help="Number of dirty data. default:50")
parser.add_argument("--dirty_file", type=str, default="train_dirty.txt", help="Path to save dirty data.")
parser.add_argument("--rest_file", type=str, default="train_dirty_rest.txt", help="The path of rest data.")
parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name")
parser.add_argument("--dirty_threshold", type=float, default="0", help="The threshold to select dirty data.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """
    Set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_local_dataset(path):
    """
    Read dataset file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sentence, label = line.strip().split("\t")
            yield {"text": sentence, "label": label}


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Preprocess dataset
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    return result


def get_dirty_data(weight_matrix, dirty_num, threshold=0):
    """
    Get index of dirty data from train data
    """
    scores = []
    for idx in range(weight_matrix.shape[0]):
        weight_sum = 0
        count = 0
        for weight in weight_matrix[idx].numpy():
            if weight > threshold:
                count += 1
                weight_sum += weight
        scores.append((count, weight_sum))
    sorted_scores = sorted(scores)[::-1]
    sorted_idxs = sorted(range(len(scores)), key=lambda idx: scores[idx])[::-1]

    ret_scores = sorted_scores[:dirty_num]
    ret_idxs = sorted_idxs[:dirty_num]

    return ret_idxs, ret_scores


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def run():
    """
    Get dirty data
    """
    set_seed(args.seed)
    paddle.set_device(args.device)
    # Define model & tokenizer
    if os.path.exists(args.params_path):
        model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))
    # Prepare & preprocess dataset
    train_path = os.path.join(args.dataset_dir, args.train_file)
    train_ds = load_dataset(read_local_dataset, path=train_path, lazy=False)

    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func)

    # Batchify dataset
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds, batch_size=args.batch_size, shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)

    # Classifier_layer_name is the layer name of the last output layer
    rep_point = RepresenterPointModel(model, train_data_loader, classifier_layer_name="classifier")
    weight_matrix = rep_point.weight_matrix

    # Save dirty data & rest data
    dirty_indexs, _ = get_dirty_data(weight_matrix, args.dirty_num, args.dirty_threshold)

    dirty_path = os.path.join(args.dataset_dir, args.dirty_file)
    rest_path = os.path.join(args.dataset_dir, args.rest_file)

    with open(dirty_path, "w") as f1, open(rest_path, "w") as f2:
        for idx in range(len(train_ds)):
            if idx in dirty_indexs:
                f1.write(train_ds.data[idx]["text"] + "\t" + train_ds.data[idx]["label"] + "\n")
            else:
                f2.write(train_ds.data[idx]["text"] + "\t" + train_ds.data[idx]["label"] + "\n")

    f1.close(), f2.close()


if __name__ == "__main__":
    run()
