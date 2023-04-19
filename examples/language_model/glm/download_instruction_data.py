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
import os
import random


def split_data(file_name):
    data = []
    with open(f"data_dir/{file_name}.json") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    print(len(data))
    random.shuffle(data)
    dev_rate = 0.01
    dev_num = int(len(data) * dev_rate)
    train_data = data[:-dev_num]
    dev_data = data[-dev_num:]
    print(len(train_data), len(dev_data))
    with open(f"data_dir/{file_name}.train.json", "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"data_dir/{file_name}.dev.json", "w") as f:
        for item in dev_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--instruction_data_name",
        default="school_math_0.25M",
        type=str,
        # required=True,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--data_path",
        default="data_dir",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.data_path, exist_ok=True)
    if not os.path.exists(f"{args.data_path}/{args.instruction_data_name}.json"):
        os.system(
            f"wget -P {args.data_path}/ https://huggingface.co/datasets/BelleGroup/{args.instruction_data_name}/resolve/main/{args.instruction_data_name}.json"
        )
        split_data(f"{args.instruction_data_name}")


if __name__ == "__main__":
    main()
