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

import json
import os
from typing import Optional

from argument import DataArgument
from data import DataFormatError, tokenize_example

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def calculate_tokens(
    dataset_name_or_path: str,
    model_name_or_path: str,
    src_length: Optional[int] = 1024,
    max_length: Optional[int] = 2048,
    num_train_epochs: Optional[int] = 3,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if (
        os.path.exists(os.path.join(dataset_name_or_path, "train.json"))
        and os.path.exists(os.path.join(dataset_name_or_path, "dev.json"))
        and os.path.exists(os.path.join(dataset_name_or_path, "test.json"))
    ):
        train_ds = load_dataset(read_local_dataset, path=os.path.join(dataset_name_or_path, "train.json"), lazy=False)
        dev_ds = load_dataset(read_local_dataset, path=os.path.join(dataset_name_or_path, "dev.json"), lazy=False)
        test_ds = load_dataset(read_local_dataset, path=os.path.join(dataset_name_or_path, "test.json"), lazy=False)
    else:
        raise DataFormatError(
            f"Unrecognized dataset name `{dataset_name_or_path}`. We expect `train.json`,`dev.json` and `test.json` under `{dataset_name_or_path}`."
        )

    data_args = DataArgument(src_length=src_length, max_length=max_length)
    epoch_tokens = 0
    for dataset in [train_ds, dev_ds, test_ds]:
        for example in dataset:
            tokenized_source, _ = tokenize_example(tokenizer, example, data_args)
            epoch_tokens += len(tokenized_source["input_ids"])

    return {"total_train_tokens": epoch_tokens * num_train_epochs, "train_tokens_per_epoch": epoch_tokens}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_or_path", type=str, help="Name or path for dataset")
    parser.add_argument("--model_name_or_path", type=str, help="Name or path for model")
    parser.add_argument("--src_length", default=1024, help="The maximum length of source(context) tokens.")
    parser.add_argument(
        "--max_length",
        default=2048,
        help="The maximum length that model input tokens can have. When intokens is set to True, it's also the maximum length for InTokens data stream",
    )
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of trainning epochs")
    args = parser.parse_args()

    output = calculate_tokens(
        args.dataset_name_or_path, args.model_name_or_path, args.src_length, args.max_length, args.num_train_epochs
    )
    print(output)
