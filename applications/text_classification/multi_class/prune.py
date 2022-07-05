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
import yaml
import functools
from typing import Optional
import paddle
import json

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, Trainer
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from paddlenlp.utils.log import logger
from dataclasses import dataclass, field

from utils import preprocess_function, read_local_dataset
from prune_trainer import DynabertConfig


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(
        default="cblue",
        metadata={"help": "Dataset for multi label classfication."})

    task_name: str = field(
        default="KUAKE-QIC",
        metadata={"help": "Task name for multi label classfication dataset."})

    dataset_dir: str = field(
        default=None,
        metadata={
            "help":
            "Local dataset directory should include train.tsv, dev.tsv and label.tsv."
        })

    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    params_dir: str = field(
        default='./checkpoint/model_state.pdparams',
        metadata={
            "help":
            "The output directory where the model checkpoints are written."
        })

    model_name_or_path: str = field(
        default='ernie-3.0-base-zh',
        metadata={
            "help":
            "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        })


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    f = open("prune_config.json")
    config = json.load(f)
    for arg in vars(training_args):
        if arg in config.keys():
            setattr(training_args, arg, config[arg])

    paddle.set_device(training_args.device)

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    if data_args.dataset_dir is not None:
        label_list = {}
        with open(os.path.join(data_args.dataset_dir, 'label.tsv'),
                  'r',
                  encoding='utf-8') as f:
            for i, line in enumerate(f):
                l = line.strip()
                label_list[l] = i
        train_ds = load_dataset(read_local_dataset,
                                path=os.path.join(data_args.dataset_dir,
                                                  'train.tsv'),
                                label_list=label_list,
                                lazy=False)
        dev_ds = load_dataset(read_local_dataset,
                              path=os.path.join(data_args.dataset_dir,
                                                'dev.tsv'),
                              label_list=label_list,
                              lazy=False)
    else:
        train_ds, dev_ds = load_dataset(data_args.dataset,
                                        name=data_args.task_name,
                                        splits=["train", "dev"])
        label_list = train_ds.label_list

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=len(label_list))
    model.set_dict(paddle.load(model_args.params_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=data_args.max_seq_length)
    train_dataset = train_ds.map(trans_func)
    dev_dataset = dev_ds.map(trans_func)

    # Define data collector， criterion
    data_collator = DataCollatorWithPadding(tokenizer)
    criterion = paddle.nn.CrossEntropyLoss()

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=dev_dataset,
                      tokenizer=tokenizer,
                      criterion=criterion)

    output_dir = training_args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer.prune(output_dir, prune_config=DynabertConfig(width_mult=1 / 12))


if __name__ == "__main__":
    main()
