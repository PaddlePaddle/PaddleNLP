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

import paddle

from paddlenlp.dataaug import WordDelete, WordInsert, WordSubstitute, WordSwap

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="../data/train.txt", help="Train dataset file name")
parser.add_argument("--aug_path", type=str, default="../data/aug.txt", help="Aug dataset file name")
parser.add_argument("--aug_strategy", choices=["mix", "substitute", "insert", "delete", "swap"], default='substitute', help="Select data augmentation strategy")
parser.add_argument("--aug_type", choices=["synonym", "homonym", "mlm"], default='synonym', help="Select data augmentation type for substitute and insert")
parser.add_argument("--create_n", type=int, default=2, help="Number of augmented sequences.")
parser.add_argument("--aug_percent", type=float, default=0.1, help="Percentage of augmented words in sequences.")
parser.add_argument('--device', default="gpu", help="Select which device to do data augmentation strategy, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def aug():
    """Do data augmentation"""
    if args.aug_strategy in ["mix", "substitute", "insert"] and args.aug_strategy == "mlm":
        paddle.set_device(args.device)

    if args.aug_strategy in ["substitute", "insert", "delete", "swap"]:
        if args.aug_strategy == "substitute":
            aug = WordSubstitute(args.aug_type, create_n=args.create_n, aug_percent=args.aug_percent)
        elif args.aug_strategy == "insert":
            aug = WordInsert(args.aug_type, create_n=args.create_n, aug_percent=args.aug_percent)
        elif args.aug_strategy == "delete":
            aug = WordDelete(create_n=args.create_n, aug_percent=args.aug_percent)
        elif args.aug_strategy == "swap":
            aug = WordSwap(create_n=args.create_n, aug_percent=args.aug_percent)
        with open(args.train_path, "r", encoding="utf-8") as f1, open(args.aug_path, "w", encoding="utf-8") as f2:
            for line in f1:
                s, l = line.strip().split("\t")

                augs = aug.augment(s)
                if not isinstance(augs[0], str):
                    augs = augs[0]
                for a in augs:
                    f2.write(a + "\t" + l + "\n")
        f1.close(), f2.close()
    elif args.aug_strategy in ["mix"]:
        aug = [
            WordSubstitute(args.aug_type, create_n=1, aug_percent=args.aug_percent),
            WordInsert(args.aug_type, create_n=1, aug_percent=args.aug_percent),
            WordDelete(create_n=1, aug_percent=args.aug_percent),
            WordSwap(create_n=1, aug_percent=args.aug_percent),
        ]
        count = 0
        with open(args.train_path, "r", encoding="utf-8") as f1, open(args.aug_path, "w", encoding="utf-8") as f2:
            for line in f1:
                s, l = line.strip().split("\t")

                for i in range(args.create_n):
                    i = count % len(aug)
                    augs = aug[i].augment(s)
                    count += 1
                    if not isinstance(augs[0], str):
                        augs = augs[0]
                    for a in augs:
                        f2.write(a + "\t" + l + "\n")
        f1.close(), f2.close()


if __name__ == "__main__":
    aug()
