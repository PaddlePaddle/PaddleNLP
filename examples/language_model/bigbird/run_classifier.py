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
from paddlenlp.transformers import BigBirdModel, BigBirdForTokenClassification, create_bigbird_simulated_attention_mask_list, create_bigbird_rand_mask_idx_list
from paddlenlp.transformers import BigBirdTokenizer
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
    "--data_dir",
    type=str,
    default='~/',
    help="vocab file used to tokenize text")
parser.add_argument(
    "--vocab_model_file",
    type=str,
    default='sentencepiece_gpt2.model',
    help="vocab model file used to tokenize text")
parser.add_argument(
    "--max_encoder_length",
    type=int,
    default=512,
    help="The maximum total input sequence length after SentencePiece tokenization."
)
parser.add_argument(
    "--num_layers",
    type=int,
    default=12,
    help="The number of BigBird Encoder layers")
#parser.add_argument("--attention_type", type=str, default="bigbird", help="")
parser.add_argument(
    "--nhead",
    type=int,
    default=12,
    help="number of heads when compute attention")
parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
parser.add_argument("--num_labels", type=int, default=2, help="")
parser.add_argument(
    "--warmup_steps",
    default=1000,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--num_train_steps",
    default=10000,
    type=int,
    help="Linear warmup over warmup_steps.")
parser.add_argument(
    "--lr", type=float, default=1e-5, help="Learning rate used to train.")
parser.add_argument(
    "--save_dir",
    type=str,
    default='chekpoints/',
    help="Directory to save model checkpoint")
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epoches for training.")

parser.add_argument(
    "--model_name_or_path", type=str, default="bigbird-base-uncased")
parser.add_argument("--pretrained_model", type=str, default=None)
parser.add_argument("--dim_feedforward", type=int, default=3072)
parser.add_argument("--activation", type=str, default="gelu")
parser.add_argument("--normalize_before", type=bool, default=False)
parser.add_argument("--block_size", type=int, default=16)
parser.add_argument("--window_size", type=int, default=3)
parser.add_argument("--num_rand_blocks", type=int, default=3)
parser.add_argument("--num_global_blocks", type=int, default=2)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
parser.add_argument("--max_position_embeddings", type=int, default=4096)
parser.add_argument("--type_vocab_size", type=int, default=2)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--pad_token_id", type=int, default=0)
parser.add_argument("--initializer_range", type=float, default=0.02)

args = parser.parse_args()


def create_dataloader(data_dir,
                      batch_size,
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
    tokenizer = BigBirdTokenizer.from_pretrained(args.model_name_or_path)
    train_data_loader, test_data_loader = \
            create_dataloader(args.data_dir, args.batch_size, args.max_encoder_length, tokenizer)
    bigbird = BigBirdModel.from_pretrained(args.model_name_or_path)
    model = BigBirdForTokenClassification(bigbird)
    bigbirdConfig = BigBirdModel.pretrained_init_configuration[
        args.model_name_or_path]

    # define metric
    criterion = nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # define optimizer
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(), learning_rate=args.lr, epsilon=1e-6)

    do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader, bigbirdConfig)

    do_evalute(model, criterion, metric, test_data_loader, bigbirdConfig)


def do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader, config):
    model.train()
    global_steps = 0
    softmax = nn.Softmax()
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
            # 训练耗时
            train_timer.tac()

            prob = softmax(output)
            correct = metric.compute(prob, labels)
            metric.update(correct)
            # save model
            # print loss
            if global_steps % 100 == 0:
                logger.info("global step %d, epoch: %d, loss: %f, acc %f" %
                            (global_steps, epoch, loss, metric.accumulate()))
                logger.info(
                    "Data processing spend %f s, training spend %f s" %
                    (data_process_timer.accumulate(), train_timer.accumulate()))
                data_process_timer.reset()
                train_timer.reset()
                metric.reset()

            if global_steps > args.num_train_steps:
                break

        logger.info("global step %d, epoch: %d, loss: %f, acc %f" %
                    (global_steps, epoch, loss, metric.accumulate()))
        metric.reset()


@paddle.no_grad()
def do_evalute(model, criterion, metric, test_data_loader, config):
    model.eval()
    global_steps = 0
    softmax = nn.Softmax()
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
        prob = softmax(output)
        correct = metric.compute(prob, labels)
        metric.update(correct)
        if global_steps % 1000 == 0:
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
