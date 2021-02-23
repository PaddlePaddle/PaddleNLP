# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import collections

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import BigBirdModel, BigBirdForTokenClassification, BigBirdTokenizer
from paddlenlp.transformers import create_bigbird_rand_mask_idx_list
from paddlenlp.utils.log import logger
from paddlenlp.datasets import Imdb
from paddlenlp.data import Stack

import os
import random
from functools import partial
import time

parser = argparse.ArgumentParser(__doc__)
parser.add_argument(
    "--batch_size",
    default=2,
    type=int,
    help="Batch size per GPU/CPU for training.", )
parser.add_argument(
    "--n_gpu", type=int, default=1, help="number of gpus to use, 0 for cpu.")
parser.add_argument(
    "--max_encoder_length",
    type=int,
    default=512,
    help="The maximum total input sequence length after SentencePiece tokenization."
)
parser.add_argument(
    "--num_train_steps",
    default=10000,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--lr", type=float, default=1e-5, help="Learning rate used to train.")
parser.add_argument(
    "--save_steps",
    type=int,
    default=100,
    help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument(
    "--save_dir",
    type=str,
    default='checkpoints/',
    help="Directory to save model checkpoint")
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="bigbird-base-uncased",
    help="pretraining model name or path")

args = parser.parse_args()


def create_dataloader(batch_size,
                      max_encoder_length,
                      tokenizer,
                      pad_val=0,
                      cls_token_id=65,
                      sep_token_id=66):
    def _tokenize(text):
        input_ids = [cls_token_id]
        input_ids.extend(
            tokenizer.convert_tokens_to_ids(
                tokenizer(text)[:max_encoder_length - 2]))
        input_ids.append(sep_token_id)
        input_len = len(input_ids)
        if input_len < max_encoder_length:
            input_ids.extend([pad_val] * (max_encoder_length - input_len))
        input_ids = np.array(input_ids).astype('int64')
        return input_ids

    def _collate_data(data, stack_fn=Stack()):
        num_fields = len(data[0])
        out = [None] * num_fields
        out[0] = stack_fn([_tokenize(x[0]) for x in data])
        out[1] = stack_fn([x[1] for x in data])
        return out

    def _create_dataloader(mode, tokenizer, max_encoder_length, pad_val):
        dataset = Imdb(mode=mode)
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=(mode == "train"))
        data_loader = paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=_collate_data,
            return_list=True)
        return data_loader

    train_data_loader = _create_dataloader("train", tokenizer,
                                           max_encoder_length, 0)
    test_data_loader = _create_dataloader("test", tokenizer, max_encoder_length,
                                          0)
    return train_data_loader, test_data_loader


class Timer(object):
    def __init__(self):
        self.reset()

    def tick(self):
        self._start = time.perf_counter()

    def tac(self):
        curr = time.perf_counter()
        self._accumulate += curr - self._start

    def accumulate(self):
        '''
        return second
        '''
        return self._accumulate

    def reset(self):
        self._start = -1
        self._accumulate = 0


def main():
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    tokenizer = BigBirdTokenizer.from_pretrained(args.model_name_or_path)
    train_data_loader, test_data_loader = \
            create_dataloader(args.batch_size, args.max_encoder_length, tokenizer)

    model = BigBirdForTokenClassification.from_pretrained(
        args.model_name_or_path)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    # define optimizer
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr, epsilon=1e-6)

    bigbirdConfig = BigBirdModel.pretrained_init_configuration[
        args.model_name_or_path]
    do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader, bigbirdConfig)

    do_evalute(model, criterion, metric, test_data_loader, bigbirdConfig)


def do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader, config):
    model.train()
    global_steps = 0
    data_process_timer = Timer()
    train_timer = Timer()
    for epoch in range(args.epochs):
        if global_steps > args.num_train_steps:
            break
        for step, batch in enumerate(train_data_loader):
            global_steps += 1
            (input_ids, labels) = batch
            seq_len = input_ids.shape[1]
            # create band mask
            data_process_timer.tick()

            rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                config["num_layers"], seq_len, seq_len, config["nhead"],
                config["block_size"], config["window_size"],
                config["num_global_blocks"], config["num_rand_blocks"],
                config["seed"])

            data_process_timer.tac()

            train_timer.tick()

            output = model(input_ids, None, rand_mask_idx_list)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()

            train_timer.tac()

            correct = metric.compute(output, labels)
            metric.update(correct)
            # print loss
            if global_steps % args.logging_steps == 0:
                logger.info("global step %d, epoch: %d, loss: %f, acc %f" %
                            (global_steps, epoch, loss, metric.accumulate()))
                logger.info(
                    "Data processing spend %f s, training spend %f s" %
                    (data_process_timer.accumulate(), train_timer.accumulate()))
                data_process_timer.reset()
                train_timer.reset()

            # save model
            if global_steps % args.save_steps == 0:
                output_dir = os.path.join(args.save_dir,
                                          "model_%d.pdparams" % (global_steps))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)

            if global_steps > args.num_train_steps:
                break

        logger.info("global step %d, epoch: %d, loss: %f, acc %f" %
                    (global_steps, epoch, loss, metric.accumulate()))
        metric.reset()


@paddle.no_grad()
def do_evalute(model, criterion, metric, test_data_loader, config):
    model.eval()
    global_steps = 0
    for step, batch in enumerate(test_data_loader):
        global_steps += 1
        (input_ids, labels) = batch
        seq_len = input_ids.shape[1]
        # create band mask
        rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
            config["num_layers"], seq_len, seq_len, config["nhead"],
            config["block_size"], config["window_size"],
            config["num_global_blocks"], config["num_rand_blocks"],
            config["seed"])
        output = model(input_ids, None, rand_mask_idx_list)
        loss = criterion(output, labels)
        correct = metric.compute(output, labels)
        metric.update(correct)
        if global_steps % args.logging_steps == 0:
            logger.info("global step %d, loss: %f, acc %f" %
                        (global_steps, loss, metric.accumulate()))
    logger.info("global step %d: loss: %f, acc %f" %
                (global_steps, loss, metric.accumulate()))
    metric.reset()
    model.train()


if __name__ == "__main__":
    if args.n_gpu > 1:
        dist.spawn(main, args=(), nprocs=args.n_gpu)
    else:
        main()
