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
import os
import sys
import random
import time
import copy
import yaml
from functools import partial
import distutils.util
import os.path as osp
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments, )
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer, )
from paddlenlp.utils.log import logger

sys.path.insert(0, os.path.abspath("."))
from sequence_classification import ClueTrainer, SeqTrainer
from question_answering import MrcTrainer
from token_classification import NerTrainer
from utils import (
    ALL_DATASETS,
    DataArguments,
    ModelArguments, )


def do_train():
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir
    ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(
                os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set_seed(args)
    data_args.dataset = data_args.dataset.strip()

    if data_args.dataset not in ALL_DATASETS:
        raise ValueError("Not found {}".format(data_args.dataset))

    config = ALL_DATASETS[data_args.dataset]
    for args in (model_args, data_args, training_args):
        for arg in vars(args):
            # if getattr(args, arg) is None:
            if arg in config.keys():
                setattr(args, arg, config[arg])

    training_args.per_device_train_batch_size = config["batch_size"]
    training_args.per_device_eval_batch_size = config["batch_size"]

    dataset_config = data_args.dataset.split(" ")
    all_ds = load_dataset(
        dataset_config[0],
        None if len(dataset_config) <= 1 else dataset_config[1],
        # lazy=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    data_args.label_list = getattr(all_ds['train'], "label_list", None)

    num_classes = 1 if all_ds["train"].label_list == None else len(all_ds[
        'train'].label_list)

    model = getattr(paddlenlp.transformers, config["model"]).from_pretrained(
        model_args.model_name_or_path, num_classes=num_classes)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if "SequenceClassification" in config["model"]:
        if 'clue' in data_args.dataset:
            trainer = ClueTrainer(all_ds["train"], all_ds["dev"], model,
                                  tokenizer, training_args)
        else:
            trainer = SeqTrainer(
                all_ds["train"],
                all_ds["dev"],
                model,
                tokenizer,
                data_args,
                training_args,
                test_ds=all_ds["test"])

    elif "QuestionAnswering" in config["model"]:
        trainer = MrcTrainer(all_ds["train"], all_ds["dev"], model, tokenizer,
                             training_args)
    elif 'TokenClassification' in config["model"]:
        trainer = NerTrainer(
            all_ds["train"],
            all_ds["dev"],
            model,
            tokenizer,
            training_args,
            test_ds=all_ds["test"])

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)

    test_ret = trainer.predict(trainer.test_ds)
    trainer.log_metrics("test", test_ret.metrics)

    input_spec = [
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64"),  # input_ids
        paddle.static.InputSpec(
            shape=[None, None], dtype="int64")  # segment_ids
    ]
    trainer.export_model(input_spec=input_spec, load_best_model=True)


if __name__ == "__main__":
    do_train()
