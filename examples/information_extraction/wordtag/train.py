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

import paddle
from paddle.io import DataLoader
import numpy as np
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer, LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.data import Stack, Pad, Tuple

from data import load_dataset, load_dict, convert_example
from metric import SequenceAccuracy


def parse_args():
    parser = argparse.ArgumentParser()

    # yapf: disable
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir, should contain train.json.")
    parser.add_argument("--model_dir", default="ernie-ctm", type=str, help="The pre-trained model checkpoint dir.")
    parser.add_argument("--output_dir", default="./outpout_dir", type=str, help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.", )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proportion over total steps.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument("--device", default="gpu",type=str, help="The device to select to train the model, is must be cpu/gpu/xpu.")
    # yapf: enable

    args = parser.parse_args()
    return args


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_ds = load_dataset(datafiles=('./data/train.json'))
    tags_to_idx = load_dict("./data/tags.txt")
    labels_to_idx = load_dict("./data/classifier_labels.txt")
    tokenizer = ErnieCtmTokenizer.from_pretrained(args.model_dir)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        tags_to_idx=tags_to_idx,
        labels_to_idx=labels_to_idx)
    train_ds.map(trans_func)

    ignore_label = tags_to_idx["O"]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # token_type_ids
        Stack(dtype='int64'),  # seq_len
        Pad(axis=0, pad_val=ignore_label, dtype='int64'),  # tags
        Stack(dtype='int64'),  # cls_label
    ): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    train_data_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        collate_fn=batchify_fn,
        return_list=True)

    model = ErnieCtmWordtagModel.from_pretrained(
        args.model_dir,
        num_cls_label=len(labels_to_idx),
        num_tag=len(tags_to_idx),
        ignore_index=tags_to_idx["O"])

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
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

    cls_acc = paddle.metric.Accuracy()
    seq_acc = SequenceAccuracy()
    total_loss = 0

    global_step = 0

    for epoch in range(1, args.num_train_epochs + 1):
        logger.info(f"Epoch {epoch} beginnig")
        start_time = time.time()

        for total_step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, seq_len, tags, cls_label = batch

            outputs = model(
                input_ids,
                token_type_ids,
                lengths=seq_len,
                tag_labels=tags,
                cls_label=cls_label)
            loss, seq_logits, cls_logits = outputs[0], outputs[1], outputs[2]
            loss = loss.mean()
            total_loss += loss
            loss.backward()

            optimizer.step()
            optimizer.clear_grad()
            lr_scheduler.step()

            cls_correct = cls_acc.compute(
                pred=cls_logits.reshape([-1, len(labels_to_idx)]),
                label=cls_label.reshape([-1]))
            cls_acc.update(cls_correct)
            seq_correct = seq_acc.compute(
                pred=seq_logits.reshape([-1, len(tags_to_idx)]),
                label=tags.reshape([-1]),
                ignore_index=tags_to_idx["O"])
            seq_acc.update(seq_correct)

            if global_step % args.logging_steps == 0 and global_step != 0:
                end_time = time.time()
                speed = float(args.logging_steps) / (end_time - start_time)
                logger.info(
                    "[Training]["
                    "epoch: %s/%s][step: %s/%s] loss: %6f, Classification Accuracy: %6f, Sequence Labeling Accuracy: %6f, speed: %6f"
                    % (epoch, args.num_train_epochs, global_step,
                       num_training_steps, total_loss / args.logging_steps,
                       cls_acc.accumulate(), seq_acc.accumulate(), speed))
                start_time = time.time()
                cls_acc.reset()
                seq_acc.reset()
                total_loss = 0

            if (global_step % args.save_steps == 0 or global_step ==
                    num_training_steps) and paddle.distributed.get_rank() == 0:
                output_dir = os.path.join(args.output_dir,
                                          "ernie_ctm_ft_model_%d.pdparams" %
                                          (global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model._layers if isinstance(
                    model, paddle.DataParallel) else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)


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
