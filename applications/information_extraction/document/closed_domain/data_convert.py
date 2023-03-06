# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import math
import os
import random
from decimal import Decimal

import numpy as np
from tqdm import tqdm
from utils import set_seed

from paddlenlp import Taskflow
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import DataConverter


def do_data_distill():
    set_seed(args.seed)

    data_converter = DataConverter(
        args.label_studio_file,
        layout_analysis=args.layout_analysis,
        schema_lang=args.schema_lang,
        ocr_lang=args.ocr_lang,
        anno_type="image",
    )

    # Generate closed-domain label maps
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    label_maps = data_converter.schema2label_maps(args.schema)
    label_maps_path = os.path.join(args.save_dir, "label_maps.json")

    # Save closed-domain label maps file
    with open(label_maps_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(label_maps, ensure_ascii=False))

    with open(args.label_studio_file, "r", encoding="utf-8") as fp:
        json_lines = json.loads(fp.read())

    # Load doccano file and convert to closed-domain data format
    if os.path.exists(args.sample_index_file):
        # Generate data with data distillation
        sample_index = json.loads(open(args.sample_index_file, "r", encoding="utf-8").readline())

        train_ids = sample_index["train_ids"]
        dev_ids = sample_index["dev_ids"]
        test_ids = sample_index["test_ids"]
    else:
        # Generate data directly from label_studio.json
        if len(args.splits) != 0 and len(args.splits) != 3:
            raise ValueError("Only []/ len(splits)==3 accepted for splits.")

        def _check_sum(splits):
            return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(str(splits[2])) == Decimal("1")

        if len(args.splits) == 3 and not _check_sum(args.splits):
            raise ValueError("Please set correct splits, sum of elements in splits should be equal to 1.")

        indexes = np.random.permutation(len(json_lines))
        index_list = indexes.tolist()
        json_lines = [json_lines[i] for i in indexes]

        i1, i2, _ = args.splits
        p1 = int(len(json_lines) * i1)
        p2 = int(len(json_lines) * (i1 + i2))

        train_ids = index_list[:p1]
        dev_ids = index_list[p1:p2]
        test_ids = index_list[p2:]

    train_lines = [json_lines[i] for i in train_ids]
    train_lines = data_converter.label_studio_to_closed_domain(train_lines, label_maps)

    dev_lines = [json_lines[i] for i in dev_ids]
    dev_lines = data_converter.label_studio_to_closed_domain(dev_lines, label_maps)

    test_lines = [json_lines[i] for i in test_ids]
    test_lines = data_converter.label_studio_to_closed_domain(test_lines, label_maps)

    if args.synthetic_ratio > 0 and os.path.exists(args.unlabeled_images_path):
        # Load trained UIE model
        uie = Taskflow("information_extraction", schema=args.schema, task_path=args.model_path)

        files_path = [
            os.path.join(args.unlabeled_images_path, file) for file in os.listdir(args.unlabeled_images_path)
        ]

        actual_ratio = math.ceil(len(files_path) / len(train_lines))
        if actual_ratio <= args.synthetic_ratio or args.synthetic_ratio == -1:
            infer_files = files_path
        else:
            idxs = random.sample(range(0, len(files_path)), args.synthetic_ratio * len(train_lines))
            infer_files = [files_path[i] for i in idxs]

        infer_results = []
        for infer_file in tqdm(infer_files, desc="Predicting: ", leave=False):
            infer_results.extend(uie({"doc": infer_file}))

        train_synthetic_lines = data_converter.uie_to_closed_domain(files_path, infer_results)

        # Concat origin and synthetic data
        train_lines.extend(train_synthetic_lines)

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    _save_examples(args.save_dir, "train_data.json", train_lines)
    _save_examples(args.save_dir, "dev_data.json", dev_lines)
    _save_examples(args.save_dir, "test_data.json", test_lines)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_studio_file", default="./data/label_studio.json", type=str, help="The annotation file exported from label studio platform.")
    parser.add_argument("--sample_index_file", default="./data/sample_index.json", type=str, help="File that record the sample index of annotation data.")
    parser.add_argument("--unlabeled_images_path", default="./data/unlabeled_images", type=str, help="File path of unlabeled images.")
    parser.add_argument("--model_path", type=str, default="../checkpoint/model_best", help="The path of saved model that you want to load.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--splits", default=[0.8, 0.1, 0.1], type=float, nargs="*", help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--synthetic_ratio", default=10, type=int, help="The ratio of labeled and synthetic samples.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--layout_analysis", default=False, type=bool, help="Enable layout analysis to optimize the order of OCR result.")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch", help="Select the language type for schema.")
    parser.add_argument("--ocr_lang", choices=["ch", "en"], default="ch", help="Select the language type for OCR.")

    args = parser.parse_args()
    # yapf: enable

    # Define your schema here
    schema = ["开票日期", "名称", "纳税人识别号", "开户行及账号", "金额", "价税合计", "No", "税率", "地址、电话", "税额", "开票人"]

    args.schema = schema

    do_data_distill()
