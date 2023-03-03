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
import math
import os
import random

from tqdm import tqdm
from utils import set_seed

from paddlenlp import Taskflow
from paddlenlp.utils.log import logger
from paddlenlp.utils.tools import DataConverter


def data_convert():
    set_seed(args.seed)

    data_converter = DataConverter(
        os.path.join(args.data_path, "label_studio.json"),
        schema_lang=args.schema_lang,
        anno_type="text",
    )

    # Generate closed-domain label maps
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    label_maps = data_converter.schema2label_maps(args.schema)
    label_maps_path = os.path.join(args.save_dir, "label_maps.json")

    # Save closed-domain label maps file
    with open(label_maps_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(label_maps, ensure_ascii=False))

    # Load doccano file and convert to distill format
    sample_index = json.loads(
        open(os.path.join(args.data_path, "sample_index.json"), "r", encoding="utf-8").readline()
    )

    train_ids = sample_index["train_ids"]
    dev_ids = sample_index["dev_ids"]
    test_ids = sample_index["test_ids"]

    with open(os.path.join(args.data_path, "label_studio.json"), "r", encoding="utf-8") as fp:
        json_lines = json.loads(fp.read())

    train_lines = [json_lines[i] for i in train_ids]
    train_lines = data_converter.label_studio_to_closed_domain(train_lines, label_maps)

    dev_lines = [json_lines[i] for i in dev_ids]
    dev_lines = data_converter.label_studio_to_closed_domain(dev_lines, label_maps)

    test_lines = [json_lines[i] for i in test_ids]
    test_lines = data_converter.label_studio_to_closed_domain(test_lines, label_maps)

    if args.synthetic_ratio > 0:
        # Load trained UIE model
        uie = Taskflow("information_extraction", schema=args.schema, task_path=args.model_path)

        # Generate synthetic data
        texts = open(os.path.join(args.data_path, "unlabeled_data.txt"), "r", encoding="utf-8").readlines()

        actual_ratio = math.ceil(len(texts) / len(train_lines))
        if actual_ratio <= args.synthetic_ratio or args.synthetic_ratio == -1:
            infer_texts = texts
        else:
            idxs = random.sample(range(0, len(texts)), args.synthetic_ratio * len(train_lines))
            infer_texts = [texts[i] for i in idxs]

        infer_results = []
        for text in tqdm(infer_texts, desc="Predicting: ", leave=False):
            infer_results.extend(uie(text))

        train_synthetic_lines = data_converter.uie_to_closed_domain(infer_texts, infer_results)

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

    parser.add_argument("--data_path", default="../data", type=str, help="The directory for labeled data with doccano format and the large scale unlabeled data.")
    parser.add_argument("--model_path", type=str, default="../checkpoint/model_best", help="The path of saved model that you want to load.")
    parser.add_argument("--save_dir", default="./data", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--synthetic_ratio", default=0, type=int, help="The ratio of labeled and synthetic samples.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization")
    parser.add_argument("--schema_lang", choices=["ch", "en"], default="ch", help="Select the language type for schema.")

    args = parser.parse_args()
    # yapf: enable

    # Define your schema here
    schema = {"武器名称": ["产国", "类型", "研发单位"]}

    args.schema = schema

    data_convert()
