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
import sys
from functools import partial

import paddle
import paddle.nn as nn

from datasets import load_dataset

from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.trainer import PdArgumentParser, CompressionArguments, Trainer
from paddlenlp.transformers import AutoTokenizer, AutoModelForTokenClassification
from paddlenlp.utils.log import logger

sys.path.append("../ernie-1.0/finetune")
from utils import DataArguments, ModelArguments


def tokenize_and_align_labels(example, tokenizer, no_entity_id, max_seq_len=512):
    if example["tokens"] == []:
        tokenized_input = {
            "labels": [],
            "input_ids": [],
            "token_type_ids": [],
            "seq_len": 0,
            "length": 0,
        }
        return tokenized_input
    tokenized_input = tokenizer(
        example["tokens"],
        max_seq_len=max_seq_len,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
        return_length=True,
    )
    label_ids = example["ner_tags"]
    if len(tokenized_input["input_ids"]) - 2 < len(label_ids):
        label_ids = label_ids[: len(tokenized_input["input_ids"]) - 2]
    label_ids = [no_entity_id] + label_ids + [no_entity_id]

    label_ids += [no_entity_id] * (len(tokenized_input["input_ids"]) - len(label_ids))
    tokenized_input["labels"] = label_ids
    return tokenized_input


def ner_trans_fn(example, tokenizer, args):
    return tokenize_and_align_labels(
        example, tokenizer=tokenizer, no_entity_id=args.no_entity_id, max_seq_len=args.max_seq_length
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    paddle.set_device(compression_args.device)

    data_args.dataset = data_args.dataset.strip()

    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    dataset_config = data_args.dataset.split(" ")
    raw_datasets = load_dataset(
        dataset_config[0],
        None if len(dataset_config) <= 1 else dataset_config[1],
    )

    label_list = raw_datasets["train"].features["ner_tags"].feature.names
    data_args.label_list = label_list
    data_args.ignore_label = -100

    data_args.no_entity_id = 0
    num_classes = 1 if label_list == None else len(label_list)

    # Define tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)

    class criterion(nn.Layer):
        def __init__(self):
            super(criterion, self).__init__()
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=data_args.ignore_label)

        def forward(self, *args, **kwargs):
            return paddle.mean(self.loss_fn(*args, **kwargs))

    loss_fct = criterion()

    # Define dataset pre-process function
    trans_fn = partial(ner_trans_fn, tokenizer=tokenizer, args=data_args)

    # Define data collector
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=data_args.ignore_label)

    column_names = raw_datasets["train"].column_names

    # Dataset pre-process
    train_dataset = raw_datasets["train"].map(trans_fn, remove_columns=column_names)
    train_dataset.label_list = label_list

    eval_dataset = raw_datasets["test"].map(trans_fn, remove_columns=column_names)
    trainer = Trainer(
        model=model,
        criterion=loss_fct,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
