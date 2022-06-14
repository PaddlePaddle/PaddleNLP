# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
import os
import random
import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    default='wiki',
    type=str,
    required=False,
    help=
    "The output directory where the model predictions and checkpoints will be written."
)
parser.add_argument("--dataset_name",
                    default='wikipedia',
                    type=str,
                    required=False,
                    help="dataset name")
parser.add_argument("--dataset_config_name",
                    default='20200501.en',
                    type=str,
                    required=False,
                    help="dataset config name")
parser.add_argument(
    "--use_slow_tokenizer",
    action="store_true",
    help=
    "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."
)
parser.add_argument("--tokenizer_name",
                    default='roberta-base',
                    type=str,
                    required=False,
                    help="tokenizer name")
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help=
    "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
)
parser.add_argument(
    "--line_by_line",
    type=bool,
    default=False,
    help=
    "Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument("--preprocessing_num_workers",
                    default=20,
                    type=int,
                    help="multi-processing number.")
parser.add_argument("--overwrite_cache",
                    type=bool,
                    default=False,
                    help="Overwrite the cached training and evaluation sets")


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets:
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)

    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset line_by_line",
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name],
                             return_special_tokens_mask=True)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: sum(examples[k], [])
                for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= args.max_seq_length:
                total_length = (total_length //
                                args.max_seq_length) * args.max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i:i + args.max_seq_length]
                    for i in range(0, total_length, args.max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {args.max_seq_length}",
        )
    tokenized_datasets.save_to_disk(args.output_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
