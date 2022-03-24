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

ALL_TASKS = {
    "SequenceClassification": [],
    "TokenClassification": [],
    "QuestionAnswering": []
}

for x in dir(paddlenlp.transformers):
    for task in ALL_TASKS.keys():
        if x.endswith(task):
            if not x.startswith("AutoModel"):
                ALL_TASKS[task].append(x)

CONFIG = yaml.load(
    open(osp.join(osp.abspath("."), "./config.yml"), 'r'),
    Loader=yaml.FullLoader)
ARGS = CONFIG["DefaultArgs"]
ALL_DATASETS = {}

for task_type in ALL_TASKS.keys():
    task = CONFIG[task_type]
    for data_name in task.keys():
        new_args = task[data_name]
        new_args = {} if new_args is None else new_args
        final_args = copy.deepcopy(ARGS)
        final_args.update(new_args)
        final_args["model"] = "AutoModelFor{}".format(task_type)
        ALL_DATASETS[data_name] = final_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to train selected in the list: " +
        ", ".join(ALL_DATASETS.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        +
        " https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html . "
        + " Such as ernie-1.0, bert-base-uncased")

    group = parser.add_argument_group(title='Common training configs.')
    group.add_argument(
        "--max_seq_length",
        default=None,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    group.add_argument(
        "--learning_rate",
        default=None,
        type=float,
        help="The initial learning rate for Adam.")
    group.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    group.add_argument(
        "--weight_decay",
        default=None,
        type=float,
        help="Weight decay if we apply some.")

    group.add_argument(
        "--num_train_epochs",
        default=None,
        type=int,
        help="Total number of training epochs to perform.", )
    group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X updates steps.")
    group.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Save checkpoint every X updates steps.")
    group.add_argument(
        "--minimum_eval_times",
        type=int,
        default=None,
        help="If under eval_steps, the valid time is less then minimum_eval_times, the config of override eval_steps."
    )
    group.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    group.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_ratio")
    group.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")

    group = parser.add_argument_group(title='Additional training configs.')
    group.add_argument(
        "--fp16",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    group.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.")
    group.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    group.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")

    group.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    group.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu"],
        help="The device to select to train the model, is must be cpu/gpu.")

    group = parser.add_argument_group(title='Additional configs for QA task.')
    group.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    group.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    group.add_argument(
        "--max_query_length", type=int, default=64, help="Max query length.")
    group.add_argument(
        "--max_answer_length", type=int, default=30, help="Max answer length.")
    group.add_argument(
        "--do_lower_case",
        action='store_false',
        help="Whether to lower case the input text. Should be True for uncased models and False for cased models."
    )

    args = parser.parse_args()
    return args


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })

    max_seq_length: int = field(
        default=128,
        metadata={
            "help":
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }, )

    # Additional configs for QA task.
    doc_stride: int = field(
        default=128,
        metadata={
            "help":
            "When splitting up a long document into chunks, how much stride to take between chunks."
        }, )

    n_best_size: int = field(
        default=20,
        metadata={
            "help":
            "The total number of n-best predictions to generate in the nbest_predictions.json output file."
        }, )

    max_query_length: int = field(
        default=64,
        metadata={"help": "Max query length."}, )

    max_answer_length: int = field(
        default=30,
        metadata={"help": "Max answer length."}, )

    do_lower_case: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to lower case the input text. Should be True for uncased models and False for cased models."
        }, )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={
        "help":
        "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
    })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def do_train():
    parser = PdArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    paddle.set_device(training_args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

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

    resume_from_checkpoint = training_args.resume_from_checkpoint
    if training_args.resume_from_checkpoint is None:
        resume_from_checkpoint = True
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    # trainer.save_infer_model() -> 部署, onnx, slim, 量化后可否加速

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # trainer.train()
    # trainer.eval()


def print_arguments(args):
    """print arguments"""
    logger.info('{:^40}'.format("Configuration Arguments"))
    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))


if __name__ == "__main__":
    # args = parse_args()

    # print_arguments(args)
    do_train()
