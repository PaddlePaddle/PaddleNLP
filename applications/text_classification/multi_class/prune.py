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
import functools

import paddle

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, Trainer, CompressionArguments
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from paddlenlp.utils.log import logger
from dataclasses import dataclass, field

from utils import preprocess_function, read_local_dataset


# yapf: disable
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_dir: str = field(default=None, metadata={"help": "Local dataset directory should include train.txt, dev.txt and label.txt."})
    max_seq_length: int = field(default=128,metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    params_dir: str = field(default='./checkpoint/',metadata={"help":"The output directory where the model checkpoints are written."})

# yapf: enable


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses(
    )
    paddle.set_device(compression_args.device)
    compression_args.strategy = 'dynabert'
    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    label_list = {}
    label_path = os.path.join(data_args.dataset_dir, 'label.txt')
    train_path = os.path.join(data_args.dataset_dir, 'train.txt')
    dev_path = os.path.join(data_args.dataset_dir, 'dev.txt')
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i

    train_ds = load_dataset(read_local_dataset,
                            path=train_path,
                            label_list=label_list,
                            lazy=False)
    dev_ds = load_dataset(read_local_dataset,
                          path=dev_path,
                          label_list=label_list,
                          lazy=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.params_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_args.params_dir)

    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=data_args.max_seq_length)
    train_dataset = train_ds.map(trans_func)
    dev_dataset = dev_ds.map(trans_func)

    # Define data collector， criterion
    data_collator = DataCollatorWithPadding(tokenizer)
    criterion = paddle.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        criterion=criterion)  # Strategy`dynabert` needs arguments `criterion`

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
