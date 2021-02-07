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
from paddlenlp.transformers import BigBirdModel, create_bigbird_simulated_attention_mask_list, create_bigbird_rand_mask_idx_list
from paddlenlp.utils.log import logger
from paddlenlp.datasets import MapDatasetWrapper

import sentencepiece as spm
import os
import random
from functools import partial
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='~/',
        help="vocab file used to tokenize text")
    parser.add_argument(
        "--vocab_model_file",
        type=str,
        default='gpt2.model',
        help="vocab model file used to tokenize text")
    parser.add_argument(
        "--max_encoder_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after SentencePiece tokenization."
    )
    parser.add_argument(
        "--init_from_check_point",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="The number of BigBird Encoder layers")
    parser.add_argument(
        "--attention_type", type=str, default="bigbird", help="")
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
        help="number of heads when compute attention")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--hidden_size", type=int, default=768, help="")
    parser.add_argument("--num_labels", type=int, default=2, help="")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
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
        "--epochs",
        type=int,
        default=10,
        help="Number of epoches for training.")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=3072)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--normalize_before", type=bool, default=False)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--num_rand_blocks", type=int, default=3)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=4096)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    args = parser.parse_args()
    return args


def create_tokenizer(model_file):
    return spm.SentencePieceProcessor(model_file=model_file)


class ImdbDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_encoder_length=512,
                 pad_val=0):
        self.samples = []
        if input_file:
            with open(input_file, "r") as f:
                for line in f.readlines():
                    line = line.rstrip("\n")
                    sample = line.split(",", 1)
                    label = np.array(int(sample[0])).astype('int64')
                    # Add [CLS] (65) and [SEP] (66) special tokens.
                    input_ids = [65]
                    input_ids.extend(
                        tokenizer.tokenize(sample[1])[:max_encoder_length - 2])
                    input_ids.append(66)
                    input_len = len(input_ids)
                    if input_len < max_encoder_length:
                        input_ids.extend([pad_val] *
                                         (max_encoder_length - input_len))
                    input_ids = np.array(input_ids).astype('int64')
                    self.samples.append([input_ids, label])

    def split(self, rate):
        num_samples = len(self.samples)
        num_save = int(num_samples * (1 - rate))
        random.shuffle(self.samples)
        split_dataset = ImdbDataset(None, None)
        split_dataset.samples = self.samples[num_save:]
        self.samples = self.samples[:num_save]
        return split_dataset

    def __getitem__(self, index):
        # [input_ids, label]
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def create_dataloader(data_dir, batch_size, max_encoder_length, tokenizer):
    def _create_dataloader(mode,
                           tokenizer,
                           max_encoder_length,
                           pad_val,
                           split_dev=True):
        input_file = os.path.join(data_dir, mode + ".csv")
        dataset = ImdbDataset(input_file, tokenizer, max_encoder_length,
                              pad_val)
        if split_dev:
            split_dataset = dataset.split(0)

            split_batch_sampler = paddle.io.BatchSampler(
                split_dataset, batch_size=batch_size)
            split_data_loader = paddle.io.DataLoader(
                dataset=split_dataset,
                batch_sampler=split_batch_sampler,
                return_list=True)

        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=(mode == "train"))
        data_loader = paddle.io.DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, return_list=True)
        if split_dev:
            return data_loader, split_data_loader
        return data_loader

    train_data_loader = _create_dataloader("train", tokenizer,
                                           max_encoder_length, 0, False)
    test_data_loader = _create_dataloader("test", tokenizer, max_encoder_length,
                                          0, False)
    return train_data_loader, test_data_loader


class ClassifierModel(nn.Layer):
    def __init__(self, num_labels, **kwargv):
        super(ClassifierModel, self).__init__()
        self.bigbird = BigBirdModel(**kwargv)
        self.linear = nn.Linear(kwargv['hidden_size'], num_labels)
        self.dropout = nn.Dropout(
            kwargv['hidden_dropout_prob'], mode="upscale_in_train")
        self.kwargv = kwargv

    def forward(self,
                input_ids,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        _, pooled_output = self.bigbird(
            input_ids,
            None,
            attention_mask_list=attention_mask_list,
            rand_mask_idx_list=rand_mask_idx_list)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output


def get_config(args, vocab_size):

    bertConfig = {
        "num_layers": args.num_layers,
        "vocab_size": vocab_size,
        "nhead": args.num_attention_heads,
        "attn_dropout": args.attn_dropout,
        "dim_feedforward": args.dim_feedforward,
        "activation": args.activation,
        "normalize_before": args.normalize_before,
        "attention_type": args.attention_type,
        "block_size": args.block_size,
        "window_size": args.window_size,
        "num_global_blocks": 2,
        "num_rand_blocks": args.num_rand_blocks,
        "seed": None,
        "pad_token_id": 0,
        "hidden_size": args.hidden_size,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "max_position_embeddings": args.max_position_embeddings,
        "type_vocab_size": args.type_vocab_size
    }
    return bertConfig


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


def main(args):
    # get dataloader
    tokenizer = create_tokenizer(args.vocab_model_file)
    vocab_size = tokenizer.vocab_size()
    bertConfig = get_config(args, vocab_size)

    train_data_loader, test_data_loader = \
            create_dataloader(args.data_dir, args.batch_size, args.max_encoder_length, tokenizer)

    # define model
    model = ClassifierModel(args.num_labels, **bertConfig)
    if args.pretrained_model is not None:
        pretrained_model_dict = paddle.load(args.pretrained_model)
        model.set_state_dict(pretrained_model_dict)
        logger.info("Pretrained model has been loaded.")

    # define metric
    criterion = nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # define optimizer

    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(args.lr,
                                                       args.num_train_steps, 0)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        lr_scheduler, args.warmup_steps, start_lr=0, end_lr=args.lr)

    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=args.weight_decay)
    logger.info("TRAIN")

    do_train(args, model, criterion, metric, optimizer, lr_scheduler,
             train_data_loader, test_data_loader, bertConfig)
    logger.info("EVAL")
    do_evalute(args, model, criterion, metric, test_data_loader, bertConfig)


def do_train(args, model, criterion, metric, optimizer, lr_scheduler,
             train_data_loader, test_data_loader, config):
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
            if args.attention_type == "bigbird":
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
            else:
                data_process_timer.tick()
                attention_mask_list, rand_mask_idx_list = create_bigbird_simulated_attention_mask_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"],
                    config["num_global_blocks"], config["num_rand_blocks"],
                    config["seed"])
                data_process_timer.tac()

                train_timer.tick()
                output = model(input_ids, attention_mask_list,
                               rand_mask_idx_list)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
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
def do_evalute(args, model, criterion, metric, test_data_loader, config):
    model.eval()
    global_steps = 0
    softmax = nn.Softmax()
    for step, batch in enumerate(test_data_loader):
        global_steps += 1
        (input_ids, labels) = batch
        seq_len = input_ids.shape[1]
        if args.attention_type == "bigbird":
            # create band mask
            rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
                config["num_layers"], seq_len, seq_len, config["nhead"],
                config["block_size"], config["window_size"],
                config["num_global_blocks"], config["num_rand_blocks"],
                config["seed"])
            output = model(input_ids, None, rand_mask_idx_list)
        else:
            attention_mask_list, rand_mask_idx_list = create_bigbird_simulated_attention_mask_list(
                config["num_layers"], seq_len, seq_len, config["nhead"],
                config["block_size"], config["window_size"],
                config["num_global_blocks"], config["num_rand_blocks"],
                config["seed"])
            output = model(input_ids, attention_mask_list, rand_mask_idx_list)
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
    args = parse_args()
    if args.n_gpu > 1:
        dist.spawn(main, args=(args, ), nprocs=args.n_gpu)
    else:
        main(args)
