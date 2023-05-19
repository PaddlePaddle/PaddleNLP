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

import argparse
import json
import os
import random
import time
from decimal import Decimal

import numpy as np
import paddle

from paddlenlp.utils.log import logger


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class LabelStudioDataConverter(object):
    """
    DataConverter to convert data export from LabelStudio platform
    """

    def __init__(self, options, text_separator):
        super().__init__()
        if isinstance(options, list) and len(options) == 1 and os.path.isfile(options[0]):
            with open(options[0], "r", encoding="utf-8") as fp:
                self.options = [x.strip() for x in fp]
        elif isinstance(options, list) and len(options) > 0:
            self.options = options
        else:
            raise ValueError(
                "Invalid options. Please use file with one label per line or set `options` with condidate labels."
            )
        self.text_separator = text_separator

    def convert_utc_examples(self, raw_examples):
        utc_examples = []
        for example in raw_examples:
            raw_text = example["data"]["text"].split(self.text_separator)
            if len(raw_text) < 1:
                continue
            elif len(raw_text) == 1:
                raw_text.append("")
            elif len(raw_text) > 2:
                raw_text = ["".join(raw_text[:-1]), raw_text[-1]]

            label_list = []
            for raw_label in example["annotations"][0]["result"][0]["value"]["choices"]:
                if raw_label not in self.options:
                    raise ValueError(
                        f"Label `{raw_label}` not found in label candidates `options`. Please recheck the data."
                    )
                label_list.append(np.where(np.array(self.options) == raw_label)[0].tolist()[0])

            utc_examples.append(
                {
                    "text_a": raw_text[0],
                    "text_b": raw_text[1],
                    "question": "",
                    "choices": self.options,
                    "labels": label_list,
                }
            )
        return utc_examples


def do_convert():
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.label_studio_file):
        raise ValueError("Please input the correct path of label studio file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError("Please set correct splits, sum of elements in splits should be equal to 1.")

    with open(args.label_studio_file, "r", encoding="utf-8") as f:
        raw_examples = json.loads(f.read())

    if args.is_shuffle:
        indexes = np.random.permutation(len(raw_examples))
        index_list = indexes.tolist()
        raw_examples = [raw_examples[i] for i in indexes]

    i1, i2, _ = args.splits
    p1 = int(len(raw_examples) * i1)
    p2 = int(len(raw_examples) * (i1 + i2))

    train_ids = index_list[:p1]
    dev_ids = index_list[p1:p2]
    test_ids = index_list[p2:]

    with open(os.path.join(args.save_dir, "sample_index.json"), "w") as fp:
        maps = {"train_ids": train_ids, "dev_ids": dev_ids, "test_ids": test_ids}
        fp.write(json.dumps(maps))

    data_converter = LabelStudioDataConverter(args.options, args.text_separator)

    train_examples = data_converter.convert_utc_examples(raw_examples[:p1])
    dev_examples = data_converter.convert_utc_examples(raw_examples[p1:p2])
    test_examples = data_converter.convert_utc_examples(raw_examples[p2:])

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    _save_examples(args.save_dir, "train.txt", train_examples)
    _save_examples(args.save_dir, "dev.txt", dev_examples)
    _save_examples(args.save_dir, "test.txt", test_examples)

    logger.info("Finished! It takes %.2f seconds" % (time.time() - tic_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_studio_file", default="./data/label_studio.json", type=str, help="The annotation file exported from label studio platform.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--text_separator", type=str, default='\t', help="Separator for classification with two input texts.")
    parser.add_argument("--options", default=None, type=str, nargs="+", help="The options for classification.")
    parser.add_argument("--is_shuffle", default=True, type=bool, help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    args = parser.parse_args()
    # yapf: enable

    do_convert()
