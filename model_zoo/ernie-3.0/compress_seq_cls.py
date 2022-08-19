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

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, Trainer, CompressionArguments
from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification
from paddlenlp.utils.log import logger

sys.path.append("../ernie-1.0/finetune")
from sequence_classification import seq_trans_fn, clue_trans_fn
from utils import ALL_DATASETS, DataArguments, ModelArguments


def main():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses(
    )

    paddle.set_device(compression_args.device)

    data_args.dataset = data_args.dataset.strip()

    if data_args.dataset in ALL_DATASETS:
        # If you custom you hyper-parameters in yaml config, it will overwrite all args.
        config = ALL_DATASETS[data_args.dataset]
        logger.info("Over-writing compression config by yaml config!")
        for args in (model_args, data_args, compression_args):
            for arg in vars(args):
                if arg in config.keys():
                    setattr(args, arg, config[arg])

        compression_args.per_device_train_batch_size = config["batch_size"]
        compression_args.per_device_eval_batch_size = config["batch_size"]

    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    dataset_config = data_args.dataset.split(" ")
    raw_datasets = load_dataset(
        dataset_config[0],
        None if len(dataset_config) <= 1 else dataset_config[1],
        splits=("train", "dev", "test"))

    data_args.label_list = getattr(raw_datasets['train'], "label_list", None)
    num_classes = 1 if raw_datasets["train"].label_list == None else len(
        raw_datasets['train'].label_list)

    criterion = paddle.nn.CrossEntropyLoss()
    # Defines tokenizer, model, loss function.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=num_classes)

    # Defines dataset pre-process function
    if "clue" in data_args.dataset:
        trans_fn = partial(clue_trans_fn, tokenizer=tokenizer, args=data_args)
    else:
        trans_fn = partial(seq_trans_fn, tokenizer=tokenizer, args=data_args)

    # Defines data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataset = raw_datasets["train"].map(trans_fn)
    eval_dataset = raw_datasets["dev"].map(trans_fn)

    trainer = Trainer(
        model=model,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        criterion=criterion)  # Strategy`dynabert` needs arguments `criterion`

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
