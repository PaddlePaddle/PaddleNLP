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

import os
import time
import argparse
import json
import random
from decimal import Decimal

from tqdm import tqdm
import numpy as np

import paddle
from paddlenlp.utils.log import logger

parser = argparse.ArgumentParser()

parser.add_argument("--doccano_file",
                    default="doccano.jsonl",
                    type=str,
                    help="The doccano file exported from doccano platform.")
parser.add_argument("--save_dir",
                    default="./data",
                    type=str,
                    help="The path of data that you wanna save.")
parser.add_argument("--splits",
                    default=[0.8, 0.1, 0.1],
                    type=float,
                    nargs="*",
                    help="The ratio of samples in datasets. "
                    "[0.8, 0.1, 0.1] means 80% samples "
                    "used for training, 10% for evaluation"
                    "and 10% for test.")
parser.add_argument("--task_type",
                    choices=['multi_class', 'multi_label', 'hierarchical'],
                    default="multi_label",
                    type=str,
                    help="Select task type, multi_class for"
                    "multi classification task, multi_label"
                    "for multi label classification task and"
                    "hierarchical for hierarchical classification,"
                    "defaults to multi_label.")
parser.add_argument("--is_shuffle",
                    default=True,
                    type=bool,
                    help="Whether to shuffle the labeled"
                    "dataset, defaults to True.")
parser.add_argument("--seed",
                    type=int,
                    default=3,
                    help="Random seed for initialization")
parser.add_argument("--separator",
                    type=str,
                    default="##",
                    help="Separator for hierarchical classification")

args = parser.parse_args()


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def do_convert():
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 2 and len(args.splits) != 3:
        raise ValueError(
            "Only len(splits)==2 / len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        if len(splits) == 2:
            return Decimal(str(splits[0])) + Decimal(str(
                splits[1])) == Decimal("1")
        if len(splits) == 3:
            return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(
                str(splits[2])) == Decimal("1")

    if not _check_sum(args.splits):
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    with open(args.doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()
    f.close()

    examples = []
    label_list = []
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            # Compatible with doccano >= 1.6.2
            if "data" in items.keys():
                text, labels = items["data"], items["label"]
            else:
                text, labels = items["text"], items["label"]
            labels = list(set(labels))
            for l in labels:
                if ',' in l:
                    raise ValueError("There exists comma \',\' in {}".format(l))

            if args.task_type == 'multi_label' or args.task_type == 'multi_class':
                example = ' '.join(
                    text.strip().split('\t')) + '\t' + ','.join(labels) + '\n'
                for l in labels:
                    if l not in label_list:
                        label_list.append(l)
            if args.task_type == 'hierarchical':
                label_dict = {}
                for label in labels:
                    for i, l in enumerate(label.split(args.separator)):
                        if i in label_dict and l not in label_dict[i]:
                            label_dict[i].append(l)
                        else:
                            label_dict[i] = [l]
                    for i in range(len(label.split(args.separator))):
                        if args.separator.join(
                                label.split(
                                    args.separator)[:i + 1]) not in label_list:
                            label_list.append(
                                args.separator.join(
                                    label.split(args.separator)[:i + 1]))
                example = ' '.join(text.strip().split('\t'))
                for i in range(len(label_dict)):
                    example += '\t' + ','.join(label_dict[i])
                example += '\n'
            examples.append(example)

    save_path = os.path.join(args.save_dir, 'label.txt')
    with open(save_path, "w", encoding="utf-8") as f:
        for l in label_list:
            f.write(l + '\n')

    def _save_examples(save_dir, file_name, examples, is_data=False):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                if is_data:
                    f.write(example.split('\t')[0] + '\n')
                else:
                    f.write(example)
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    if args.is_shuffle:
        indexes = np.random.permutation(len(raw_examples))
        raw_examples = [raw_examples[i] for i in indexes]
    if len(args.splits) == 2:
        i1, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        _save_examples(args.save_dir, "train.txt", examples[:p1])
        _save_examples(args.save_dir, "dev.txt", examples[p1:])
        _save_examples(args.save_dir, "data.txt", examples[p1:], True)
    if len(args.splits) == 3:
        i1, i2, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))
        _save_examples(args.save_dir, "train.txt", examples[:p1])
        _save_examples(args.save_dir, "dev.txt", examples[p1:p2])
        _save_examples(args.save_dir, "test.txt", examples[p2:])
        _save_examples(args.save_dir, "data.txt", examples[p2:], True)
    logger.info('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == '__main__':
    do_convert()
