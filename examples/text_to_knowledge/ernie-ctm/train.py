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

import os
import argparse
import time
import random
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer, LinearDecayWithWarmup
from paddlenlp.data import Stack, Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.layers.crf import LinearChainCrf, LinearChainCrfLoss

from data import load_dict, convert_example, read_custom_data, create_dataloader
from metric import SequenceAccuracy


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain train.json.")
    parser.add_argument("--init_from_ckpt", default=None, type=str, help="The path of checkpoint to be loaded.")
    parser.add_argument("--output_dir", default="./output", type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proportion over total steps.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=1000, type=int, help="random seed for initialization")
    parser.add_argument("--device", default="gpu",type=str, help="The device to select to train the model, is must be cpu/gpu/xpu.")
    # yapf: enable

    args = parser.parse_args()
    return args


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, metric, data_loader, tags, tags_to_idx):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader():
        input_ids, token_type_ids, seq_len, tags = batch
        loss, seq_logits = model(input_ids,
                                 token_type_ids,
                                 lengths=seq_len,
                                 tag_labels=tags)
        loss = loss.mean()
        losses.append(loss.numpy())

        correct = metric.compute(pred=seq_logits.reshape([-1,
                                                          len(tags_to_idx)]),
                                 label=tags.reshape([-1]),
                                 ignore_index=tags_to_idx["O"])
        metric.update(correct)
    acc = metric.accumulate()
    logger.info("eval loss: %.5f, acc: %.5f" % (np.mean(losses), acc))
    model.train()
    metric.reset()


def do_train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds = load_dataset(read_custom_data,
                            filename=os.path.join(args.data_dir, "train.txt"),
                            is_test=False,
                            lazy=False)
    dev_ds = load_dataset(read_custom_data,
                          filename=os.path.join(args.data_dir, "dev.txt"),
                          is_test=False,
                          lazy=False)
    tags_to_idx = load_dict(os.path.join(args.data_dir, "tags.txt"))

    tokenizer = ErnieCtmTokenizer.from_pretrained("wordtag")
    model = ErnieCtmWordtagModel.from_pretrained("wordtag",
                                                 num_tag=len(tags_to_idx))
    model.crf_loss = LinearChainCrfLoss(
        LinearChainCrf(len(tags_to_idx), 0.1, with_start_stop_tag=False))

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_len=args.max_seq_len,
                         tags_to_idx=tags_to_idx)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'
            ),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=tags_to_idx["O"], dtype='int64'),  # tags
    ): fn(samples)

    train_data_loader = create_dataloader(train_ds,
                                          mode="train",
                                          batch_size=args.batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    dev_data_loader = create_dataloader(dev_ds,
                                        mode="dev",
                                        batch_size=args.batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.num_train_epochs
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    num_train_optimization_steps = len(
        train_ds) / args.batch_size * args.num_train_epochs

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    logger.info("Total steps: %s" % num_training_steps)
    logger.info("WarmUp steps: %s" % warmup)

    metric = SequenceAccuracy()

    total_loss = 0
    global_step = 0

    for epoch in range(1, args.num_train_epochs + 1):
        logger.info(f"Epoch {epoch} beginnig")
        start_time = time.time()

        for total_step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, seq_len, tags = batch

            loss, _ = model(input_ids,
                            token_type_ids,
                            lengths=seq_len,
                            tag_labels=tags)
            loss = loss.mean()
            total_loss += loss
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

            if global_step % args.logging_steps == 0 and rank == 0:
                end_time = time.time()
                speed = float(args.logging_steps) / (end_time - start_time)
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, total_loss / args.logging_steps,
                       speed))
                start_time = time.time()
                total_loss = 0

            if (global_step % args.save_steps == 0
                    or global_step == num_training_steps) and rank == 0:
                output_dir = os.path.join(args.output_dir,
                                          "model_%d" % (global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

        evaluate(model, metric, dev_data_loader, tags, tags_to_idx)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
