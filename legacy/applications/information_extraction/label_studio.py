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

from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import DataConverter


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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

    if raw_examples[0]["data"].get("image"):
        anno_type = "image"
    else:
        anno_type = "text"

    data_converter = DataConverter(
        args.label_studio_file,
        negative_ratio=args.negative_ratio,
        prompt_prefix=args.prompt_prefix,
        options=args.options,
        separator=args.separator,
        layout_analysis=args.layout_analysis,
        schema_lang=args.schema_lang,
        ocr_lang=args.ocr_lang,
        anno_type=anno_type,
    )

    if args.task_type == "ext":
        train_examples = data_converter.convert_ext_examples(raw_examples[:p1])
        dev_examples = data_converter.convert_ext_examples(raw_examples[p1:p2], is_train=False)
        test_examples = data_converter.convert_ext_examples(raw_examples[p2:], is_train=False)
    else:
        train_examples = data_converter.convert_cls_examples(raw_examples[:p1])
        dev_examples = data_converter.convert_cls_examples(raw_examples[p1:p2])
        test_examples = data_converter.convert_cls_examples(raw_examples[p2:])

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
    parser.add_argument("--negative_ratio", default=5, type=int, help="Used only for the extraction task, the ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str, help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.")
    parser.add_argument("--options", default=["正向", "负向"], type=str, nargs="+", help="Used only for the classification task, the options for classification")
    parser.add_argument("--prompt_prefix", default="情感倾向", type=str, help="Used only for the classification task, the prompt prefix for classification")
    parser.add_argument("--is_shuffle", default="True", type=strtobool, help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--layout_analysis", default=False, type=bool, help="Enable layout analysis to optimize the order of OCR result.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--separator", type=str, default='##', help="Used only for entity/aspect-level classification task, separator for entity label and classification label")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch", help="Select the language type for schema.")
    parser.add_argument("--ocr_lang", choices=["ch", "en"], default="ch", help="Select the language type for OCR.")

    args = parser.parse_args()
    # yapf: enable

    do_convert()
