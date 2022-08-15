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


# yapf: disable
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_dir: str = field(default=None, metadata={"help": "The dataset directory should include train.txt, dev.txt and label.txt files."})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    params_dir: str = field(default='./checkpoint/', metadata={"help": "The output directory where the model checkpoints are written."})
    width_mult: str = field(default='2/3', metadata={"help": "The reserved ratio for q, k, v, and ffn weight widths."})

# yapf: enable


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    paddle.set_device(training_args.device)

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # load and preprocess dataset
    label_list = {}
    with open(os.path.join(data_args.dataset_dir, 'label.txt'),
              'r',
              encoding='utf-8') as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i
    train_ds = load_dataset(read_local_dataset,
                            path=os.path.join(data_args.dataset_dir,
                                              'train.txt'),
                            label_list=label_list,
                            lazy=False)
    dev_ds = load_dataset(read_local_dataset,
                          path=os.path.join(data_args.dataset_dir, 'dev.txt'),
                          label_list=label_list,
                          lazy=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.params_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.params_dir)

    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=data_args.max_seq_length,
                                   label_nums=len(label_list))
    train_dataset = train_ds.map(trans_func)
    dev_dataset = dev_ds.map(trans_func)
    # Define data collector， criterion
    data_collator = DataCollatorWithPadding(tokenizer)
    criterion = paddle.nn.BCEWithLogitsLoss()

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

    trainer.prune(
        output_dir,
        prune_config=DynabertConfig(width_mult=eval(model_args.width_mult)))


if __name__ == "__main__":
    main()
