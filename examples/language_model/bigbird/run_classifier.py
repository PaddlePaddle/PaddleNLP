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

import os
import random
from functools import partial
import time

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import BigBirdModel, BigBirdForSequenceClassification, BigBirdTokenizer
from paddlenlp.transformers import create_bigbird_rand_mask_idx_list
from paddlenlp.utils.log import logger
from paddlenlp.datasets import Imdb
from paddlenlp.data import Stack

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="bigbird-base-uncased", help="pretraining model name or path")
parser.add_argument("--max_encoder_length", type=int, default=3072, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate used to train.")
parser.add_argument("--max_steps", default=10000, type=int, help="Max training steps to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention ffn model dropout.")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.0, help="The dropout rate for the embedding pooler.")
parser.add_argument("--device", type=str, default="gpu", help="Select cpu, gpu, xpu devices to train model.")
# yapf: enable
args = parser.parse_args()


def create_dataloader(batch_size, max_encoder_length, tokenizer, pad_val=0):
    def _tokenize(text):
        input_ids = [tokenizer.cls_id]
        input_ids.extend(
            tokenizer.convert_tokens_to_ids(
                tokenizer._tokenize(text)[:max_encoder_length - 2]))
        input_ids.append(tokenizer.sep_id)
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
        seq_len = len(out[0][0])
        # Construct the random attention mask for the random attention
        rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
            config["num_layers"], seq_len, seq_len, config["nhead"],
            config["block_size"], config["window_size"],
            config["num_global_blocks"], config["num_rand_blocks"],
            config["seed"])
        out.extend(rand_mask_idx_list)
        return out

    def _create_dataloader(mode, tokenizer, max_encoder_length, pad_val=0):
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


def main():
    # Initialization for the parallel enviroment
    paddle.set_device(args.device)
    # Define the model and metric 
    model = BigBirdForSequenceClassification.from_pretrained(
        args.model_name_or_path)
    criterion = nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # Define the tokenizer and dataloader
    tokenizer = BigBirdTokenizer.from_pretrained(args.model_name_or_path)
    global config
    config = BigBirdModel.pretrained_init_configuration[args.model_name_or_path]
    train_data_loader, test_data_loader = \
            create_dataloader(args.batch_size, args.max_encoder_length, tokenizer)

    # Define the Adam optimizer 
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=args.learning_rate,
        epsilon=1e-6)

    # Finetune the classification model
    do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader)

    # Evaluate the finetune model
    do_evalute(model, criterion, metric, test_data_loader)


def do_train(model, criterion, metric, optimizer, train_data_loader,
             test_data_loader):
    model.train()
    global_steps = 0
    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader):
            global_steps += 1
            input_ids, labels = batch[:2]
            rand_mask_idx_list = batch[2:]

            output = model(input_ids, None, rand_mask_idx_list)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            correct = metric.compute(output, labels)
            metric.update(correct)

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "train: global step %d, epoch: %d, loss: %f, acc:%f, speed: %.2f step/s"
                    % (global_steps, epoch, loss, metric.accumulate(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_steps % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d.pdparams" % (global_steps))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)

            if global_steps >= args.max_steps:
                break
        metric.reset()


@paddle.no_grad()
def do_evalute(model, criterion, metric, test_data_loader):
    model.eval()
    global_steps = 0
    for step, batch in enumerate(test_data_loader):
        global_steps += 1
        input_ids, labels = batch[:2]
        rand_mask_idx_list = batch[2:]
        output = model(input_ids, None, rand_mask_idx_list)
        loss = criterion(output, labels)
        correct = metric.compute(output, labels)
        metric.update(correct)
        if global_steps % args.logging_steps == 0:
            logger.info("eval: global step %d, loss: %f, acc %f" %
                        (global_steps, loss, metric.accumulate()))
    logger.info("final eval: loss: %f, acc %f" % (loss, metric.accumulate()))
    metric.reset()
    model.train()


if __name__ == "__main__":
    main()
