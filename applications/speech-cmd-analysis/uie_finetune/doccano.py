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

import numpy as np

from utils import set_seed, save_examples, convert_doccano_examples


def convert_doccano_file(doccano_file,
                         save_dir,
                         splits=[0.8, 0.9],
                         negative_ratio=1,
                         is_shuffle=True):
    """
        @Description: Consvert doccano file to data format which is suitable to input to this Application.
        @Param doccano_file: The annotated file exported from doccano labeling platform.
        @Param save_dir: The directory of data that you wanna save.
        @Param splits: Whether to split doccano file into train/dev/test, note: Only []/ len(splits)==2 accepted.
        @Param negative_ratio: The ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples.
        @Param is_shuffle: Whether to shuffle data.
    """
    tic_time = time.time()
    if not os.path.exists(doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if len(splits) != 0 and len(splits) != 2:
        raise ValueError("Only []/ len(splits)==2 accepted for splits.")

    if splits and (splits[0] >= splits[1] or splits[0] >= 1.0 or
                   splits[1] >= 1.0 or splits[0] <= 0. or splits[1] <= 0):
        raise ValueError(
            "Please set correct splits, the element in it should be in (0,1), and splits[1]>splits[0]."
        )

    with open(doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()

    entity_examples, relation_examples = convert_doccano_examples(
        raw_examples, negative_ratio)

    examples = [e + r for e, r in zip(entity_examples, relation_examples)]

    # index for saving data
    idxs = np.arange(len(examples))

    if is_shuffle:
        idxs = np.random.permutation(idxs)

    if len(splits) == 0:
        save_path = os.path.join(save_dir, "doccano.txt")
        save_examples(examples, save_path, idxs)
        print(f"\nSave data to {save_path}.")
    else:
        eth1, eth2 = int(len(examples) * splits[0]), int(
            len(examples) * splits[1])
        save_train_path = os.path.join(save_dir, "train.txt")
        save_dev_path = os.path.join(save_dir, "dev.txt")
        save_test_path = os.path.join(save_dir, "test.txt")
        save_examples(examples, save_train_path, idxs[:eth1])
        save_examples(examples, save_dev_path, idxs[eth1:eth2])
        save_examples(examples, save_test_path, idxs[eth2:])
        print(f"\nSave train data to {save_train_path}.")
        print(f"Save dev data to {save_dev_path}.")
        print(f"Save test data to {save_test_path}.")

    print('Finished! It takes %.2f seconds' % (time.time() - tic_time))


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doccano_file",
        type=str,
        default="./data/doccano.json",
        help="The doccano file exported from doccano platform.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/ext_data",
        help="The path of data that you wanna save.")
    parser.add_argument(
        "--negative_ratio",
        type=int,
        default=5,
        help="The ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument(
        "--splits",
        type=float,
        nargs='*',
        default=[0.6, 0.8],
        help="The ratio of samples in datasets. [0.6, 0.8] means 60% samples used for training, 20% for evaluation and 20% for test.")
    args = parser.parse_args()
    # yapf: enable

    # Ensure generate the same negative samples for one seed.
    set_seed(1000)

    convert_doccano_file(
        args.doccano_file,
        args.save_dir,
        splits=args.splits,
        negative_ratio=args.negative_ratio,
        is_shuffle=True)
