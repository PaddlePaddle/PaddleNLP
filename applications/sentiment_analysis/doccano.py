# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import json
import argparse
import numpy as np
from utils import decoding, concate_aspect_and_opinion, save_examples, save_dict


def doccano2SA(doccano_file,
               save_ext_dir,
               save_cls_dir,
               splits=[0.8, 0.9],
               is_shuffle=True):
    """
        @Description: Consvert doccano file to data format which is suitable to input to this Application.
        @Param doccano_file: The annotated file exported from doccano labeling platform.
        @Param save_ext_dir: The directory of ext data that you wanna save.
        @Param save_cls_dir: The directory of cls data that you wanna save.
        @Param splits: Whether to split doccano file into train/dev/test, note: Only []/ len(splits)==2 accepted.
        @Param is_shuffle: Whether to shuffle data.
    """
    if not os.path.exists(doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(save_ext_dir):
        os.makedirs(save_ext_dir)

    if not os.path.exists(save_cls_dir):
        os.makedirs(save_cls_dir)

    if len(splits) != 0 and len(splits) != 2:
        raise ValueError("Only []/ len(splits)==2 accepted for splits.")

    if splits and (splits[0] >= splits[1] or splits[0] >= 1.0
                   or splits[1] >= 1.0 or splits[0] <= 0. or splits[1] <= 0):
        raise ValueError(
            "Please set correct splits, the element in it should be in (0,1), and splits[1]>splits[0]."
        )

    def label_ext_with_label_term(ext_label, start, end, tag):

        if tag == "Opinion":
            b_tag = "B-Opinion"
            i_tag = "I-Opinion"
        else:
            b_tag = "B-Aspect"
            i_tag = "I-Aspect"

        ext_label[start] = b_tag
        for i in range(start + 1, end):
            ext_label[i] = i_tag

    ext_examples, cls_examples = [], []
    with open(doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()
    # start to label for ext and cls data
    for line in raw_examples:
        items = json.loads(line)
        text, label_terms = items["data"], items["label"]
        # label ext data with label_terms
        ext_label = ["O"] * len(text)
        aspect_mapper = {}
        for label_term in label_terms:
            start, end, tag = label_term
            label_ext_with_label_term(ext_label, start, end, tag)
            if tag == "Pos-Aspect":
                aspect_mapper[text[start:end]] = "1"
            elif tag == "Neg-Aspect":
                aspect_mapper[text[start:end]] = "0"
        ext_examples.append((text, " ".join(ext_label)))
        # label cls data
        aps = decoding(text, ext_label)
        for ap in aps:
            aspect, opinions = ap[0], list(set(ap[1:]))
            if aspect not in aspect_mapper:
                continue
            aspect_text = concate_aspect_and_opinion(text, aspect, opinions)
            cls_examples.append((aspect_mapper[aspect], aspect_text, text))

    # index for saving data
    ext_idx = np.arange(len(ext_examples))
    cls_idx = np.arange(len(cls_examples))

    if is_shuffle:
        ext_idx = np.random.permutation(ext_idx)
        cls_idx = np.random.permutation(cls_idx)

    if len(splits) == 0:
        # save ext data
        save_ext_path = os.path.join(save_ext_dir, "doccano.txt")
        save_examples(ext_examples, save_ext_path, ext_idx)
        print(f"\next: save data to {save_ext_path}.")
        # save cls data
        save_cls_path = os.path.join(save_cls_dir, "doccano.txt")
        save_examples(cls_examples, save_cls_path, cls_idx)
        print(f"\ncls: save data to {save_cls_path}.")

    else:
        # save ext data
        eth1, eth2 = int(len(ext_examples) * splits[0]), int(
            len(ext_examples) * splits[1])
        save_ext_train_path = os.path.join(save_ext_dir, "train.txt")
        save_ext_dev_path = os.path.join(save_ext_dir, "dev.txt")
        save_ext_test_path = os.path.join(save_ext_dir, "test.txt")
        save_examples(ext_examples, save_ext_train_path, ext_idx[:eth1])
        save_examples(ext_examples, save_ext_dev_path, ext_idx[eth1:eth2])
        save_examples(ext_examples, save_ext_test_path, ext_idx[eth2:])
        print(f"\next: save train data to {save_ext_train_path}.")
        print(f"ext: save dev data to {save_ext_dev_path}.")
        print(f"ext: save test data to {save_ext_test_path}.")

        # save cls data
        cth1, cth2 = int(len(cls_examples) * splits[0]), int(
            len(cls_examples) * splits[1])
        save_cls_train_path = os.path.join(save_cls_dir, "train.txt")
        save_cls_dev_path = os.path.join(save_cls_dir, "dev.txt")
        save_cls_test_path = os.path.join(save_cls_dir, "test.txt")
        save_examples(cls_examples, save_cls_train_path, cls_idx[:cth1])
        save_examples(cls_examples, save_cls_dev_path, cls_idx[cth1:cth2])
        save_examples(cls_examples, save_cls_test_path, cls_idx[cth2:])
        print(f"\ncls: save train data to {save_cls_train_path}.")
        print(f"cls: save dev data to {save_cls_dev_path}.")
        print(f"cls: save test data to {save_cls_test_path}.")

    # save ext dict
    ext_dict_path = os.path.join(save_ext_dir, "label.dict")
    cls_dict_path = os.path.join(save_cls_dir, "label.dict")
    save_dict(ext_dict_path, "ext")
    save_dict(cls_dict_path, "cls")
    print(f"\next: save dict to {ext_dict_path}.")
    print(f"cls: save dict to {cls_dict_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doccano_file",
                        type=str,
                        default="./data/doccano.json",
                        help="The doccano file exported from doccano platform.")
    parser.add_argument("--save_ext_dir",
                        type=str,
                        default="./data/ext_data1",
                        help="The path of ext data that you wanna save.")
    parser.add_argument("--save_cls_dir",
                        type=str,
                        default="./data/cls_data1",
                        help="The path of cls data that you wanna save.")
    args = parser.parse_args()

    doccano2SA(args.doccano_file,
               args.save_ext_dir,
               args.save_cls_dir,
               is_shuffle=True)
