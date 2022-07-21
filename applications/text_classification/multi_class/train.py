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

import re
import json
import functools
import random
import time
import os
import argparse

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer, LinearDecayWithWarmup
from paddlenlp.utils.log import logger

from utils import evaluate, preprocess_function, read_local_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir",
                    default="./checkpoint",
                    type=str,
                    help="The output directory where the model "
                    "checkpoints will be written.")
parser.add_argument("--dataset",
                    default="cblue",
                    type=str,
                    help="Dataset for text classfication.")
parser.add_argument("--dataset_dir",
                    default=None,
                    type=str,
                    help="Local dataset directory should include"
                    "train.txt, dev.txt and label.txt")
parser.add_argument("--task_name",
                    default="KUAKE-QIC",
                    type=str,
                    help="Task name for text classfication dataset.")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length"
                    "after tokenization. Sequences longer than this "
                    "will be truncated, sequences shorter will be padded.")
parser.add_argument('--model_name',
                    default="ernie-3.0-base-zh",
                    help="Select model to train, defaults "
                    "to ernie-3.0-base-zh.")
parser.add_argument('--device',
                    choices=['cpu', 'gpu', 'xpu', 'npu'],
                    default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate",
                    default=6e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay",
                    default=0.01,
                    type=float,
                    help="Weight decay if we apply some.")
parser.add_argument('--early_stop',
                    action='store_true',
                    help='Epoch before early stop.')
parser.add_argument('--early_stop_nums',
                    type=int,
                    default=4,
                    help='Number of epoch before early stop.')
parser.add_argument("--epochs",
                    default=100,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--warmup',
                    action='store_true',
                    help="whether use warmup strategy")
parser.add_argument('--warmup_proportion',
                    default=0.1,
                    type=float,
                    help="Linear warmup proportion of learning "
                    "rate over the training process.")
parser.add_argument("--logging_steps",
                    default=5,
                    type=int,
                    help="The interval steps to logging.")
parser.add_argument("--init_from_ckpt",
                    type=str,
                    default=None,
                    help="The path of checkpoint to be loaded.")
parser.add_argument("--seed",
                    type=int,
                    default=3,
                    help="random seed for initialization")

args = parser.parse_args()


def set_seed(seed):
    """
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def args_saving():
    argsDict = args.__dict__
    with open(os.path.join(args.save_dir, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def train():
    """
    Training a binary or multi classification model
    """

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_saving()
    set_seed(args.seed)
    paddle.set_device(args.device)

    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # load and preprocess dataset
    if args.dataset_dir is not None:
        label_list = {}
        with open(os.path.join(args.dataset_dir, 'label.txt'),
                  'r',
                  encoding='utf-8') as f:
            for i, line in enumerate(f):
                l = line.strip()
                label_list[l] = i
        train_ds = load_dataset(read_local_dataset,
                                path=os.path.join(args.dataset_dir,
                                                  'train.txt'),
                                label_list=label_list,
                                lazy=False)
        dev_ds = load_dataset(read_local_dataset,
                              path=os.path.join(args.dataset_dir, 'dev.txt'),
                              label_list=label_list,
                              lazy=False)
    else:
        train_ds, dev_ds = load_dataset(args.dataset,
                                        name=args.task_name,
                                        splits=["train", "dev"])
        label_list = train_ds.label_list

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)
    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    if paddle.distributed.get_world_size() > 1:
        train_batch_sampler = DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
    else:
        train_batch_sampler = BatchSampler(train_ds,
                                           batch_size=args.batch_size,
                                           shuffle=True)
    dev_batch_sampler = BatchSampler(dev_ds,
                                     batch_size=args.batch_size,
                                     shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=collate_fn)
    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=collate_fn)

    # define model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_classes=len(label_list))
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = Accuracy()

    global_step = 0
    best_acc = 0
    early_stop_count = 0
    tic_train = time.time()

    for epoch in range(1, args.epochs + 1):

        if args.early_stop and early_stop_count >= args.early_stop_nums:
            logger.info("Early stop!")
            break

        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch['input_ids'], batch[
                'token_type_ids'], batch['labels']

            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)

            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            loss.backward()
            optimizer.step()
            if args.warmup:
                lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc, 10 /
                       (time.time() - tic_train)))
                tic_train = time.time()

        early_stop_count += 1
        acc = evaluate(model, criterion, metric, dev_data_loader)

        save_best_path = args.save_dir
        if not os.path.exists(save_best_path):
            os.makedirs(save_best_path)

        # save models
        if acc > best_acc:
            logger.info("Current best accuracy: %.5f" % (acc))
            early_stop_count = 0
            best_acc = acc
            model._layers.save_pretrained(save_best_path)
            tokenizer.save_pretrained(save_best_path)

    logger.info("Final best accuracy: %.5f" % (best_acc))
    logger.info("Save best accuracy text classification model in %s" %
                (args.save_dir))


if __name__ == "__main__":
    train()
