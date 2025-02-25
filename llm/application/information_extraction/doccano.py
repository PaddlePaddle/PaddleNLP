# coding=utf-8
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os
import time
from decimal import Decimal

import numpy as np
from utils import convert_llm_examples, set_seed

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger


def do_convert():
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError("Please set correct splits, sum of elements in splits should be equal to 1.")

    with open(args.doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()

    def _create_llm_examples(
        examples,
        negative_ratio,
        shuffle=False,
        is_train=True,
        schema_lang="ch",
    ):
        entities, relations = convert_llm_examples(examples, negative_ratio, is_train, schema_lang)
        examples = entities + relations
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    if len(args.splits) == 0:
        examples = _create_llm_examples(
            raw_examples,
            args.negative_ratio,
            args.is_shuffle,
            schema_lang=args.schema_lang,
        )

        _save_examples(args.save_dir, "train.json", examples)

    else:
        if args.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            index_list = indexes.tolist()
            raw_examples = [raw_examples[i] for i in indexes]
        else:
            index_list = list(range(len(raw_examples)))

        i1, i2, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        train_ids = index_list[:p1]
        dev_ids = index_list[p1:p2]
        test_ids = index_list[p2:]

        with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
            maps = {"train_ids": train_ids, "dev_ids": dev_ids, "test_ids": test_ids}
            fp.write(json.dumps(maps))

        train_examples = _create_llm_examples(
            raw_examples[:p1],
            args.negative_ratio,
            args.is_shuffle,
            schema_lang=args.schema_lang,
        )
        dev_examples = _create_llm_examples(
            raw_examples[p1:p2],
            -1,
            is_train=False,
            schema_lang=args.schema_lang,
        )
        test_examples = _create_llm_examples(
            raw_examples[p2:],
            -1,
            is_train=False,
            schema_lang=args.schema_lang,
        )

        _save_examples(args.save_dir, "train.json", train_examples)
        _save_examples(args.save_dir, "dev.json", dev_examples)
        _save_examples(args.save_dir, "test.json", test_examples)

    logger.info("Finished! It takes %.2f seconds" % (time.time() - tic_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--doccano_file", default="./data/doccano_ext.json", type=str, help="The doccano file exported from doccano platform.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--negative_ratio", default=5, type=int, help="Used only for the extraction task, the ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--task_type", choices="ie", default="ie", type=str, help="Select task type, ie for the information extraction task used qwen2, defaults to ie.")
    parser.add_argument("--is_shuffle", default="False", type=strtobool, help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch", help="Select the language type for schema.")

    args = parser.parse_args()
    # yapf: enable

    do_convert()
