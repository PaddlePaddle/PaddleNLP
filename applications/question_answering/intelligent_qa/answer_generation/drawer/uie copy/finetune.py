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
import time
import os
from functools import partial

import paddle
from paddle.utils.download import get_path_from_url
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.utils.log import logger

from model import UIE
from evaluate import evaluate
from utils import set_seed, convert_example, reader, MODEL_MAP


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    resource_file_urls = MODEL_MAP[args.model]['resource_file_urls']

    logger.info("Downloading resource files...")
    for key, val in resource_file_urls.items():
        file_path = os.path.join(args.model, key)
        if not os.path.exists(file_path):
            get_path_from_url(val, args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = UIE.from_pretrained(args.model)

    train_ds = load_dataset(reader,
                            data_path=args.train_path,
                            max_seq_len=args.max_seq_len,
                            lazy=False)
    dev_ds = load_dataset(reader,
                          data_path=args.dev_path,
                          max_seq_len=args.max_seq_len,
                          lazy=False)

    train_ds = train_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len))
    dev_ds = dev_ds.map(
        partial(convert_example,
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len))

    train_batch_sampler = paddle.io.BatchSampler(dataset=train_ds,
                                                 batch_size=args.batch_size,
                                                 shuffle=True)
    train_data_loader = paddle.io.DataLoader(dataset=train_ds,
                                             batch_sampler=train_batch_sampler,
                                             return_list=True)

    dev_batch_sampler = paddle.io.BatchSampler(dataset=dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    dev_data_loader = paddle.io.DataLoader(dataset=dev_ds,
                                           batch_sampler=dev_batch_sampler,
                                           return_list=True)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate,
                                       parameters=model.parameters())

    criterion = paddle.nn.BCELoss()
    metric = SpanEvaluator()

    loss_list = []
    global_step = 0
    best_step = 0
    best_f1 = 0
    tic_train = time.time()
    for epoch in range(1, args.num_epochs + 1):
        for batch in train_data_loader:
            input_ids, token_type_ids, att_mask, pos_ids, start_ids, end_ids = batch
            start_prob, end_prob = model(input_ids, token_type_ids, att_mask,
                                         pos_ids)
            start_ids = paddle.cast(start_ids, 'float32')
            end_ids = paddle.cast(end_ids, 'float32')
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            loss_list.append(float(loss))

            global_step += 1
            if global_step % args.logging_steps == 0 and rank == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg,
                       args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0 and rank == 0:
                save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(save_dir)
                logger.disable()
                tokenizer.save_pretrained(save_dir)
                logger.enable()

                precision, recall, f1 = evaluate(model, metric, dev_data_loader)
                logger.info(
                    "Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                    (precision, recall, f1))
                if f1 > best_f1:
                    logger.info(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    save_dir = os.path.join(args.save_dir, "model_best")
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(save_dir)
                    logger.disable()
                    tokenizer.save_pretrained(save_dir)
                    logger.enable()
                tic_train = time.time()


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
    parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_len", default=512, type=int, help="The maximum input sequence length. "
        "Sequences longer than this will be split automatically.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", default=1000, type=int, help="Random seed for initialization")
    parser.add_argument("--logging_steps", default=10, type=int, help="The interval steps to logging.")
    parser.add_argument("--valid_steps", default=100, type=int, help="The interval steps to evaluate model performance.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--model", choices=["uie-base", "uie-tiny", "uie-medium", "uie-mini", "uie-micro", "uie-nano"], default="uie-base", type=str, help="Select the pretrained model for few-shot learning.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of model parameters for initialization.")

    args = parser.parse_args()
    # yapf: enable

    do_train()
