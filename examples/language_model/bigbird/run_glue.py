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
import logging
import os
import sys
import random
import time
import math
import distutils.util
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BigBirdModel, BigBirdForSequenceClassification, BigBirdTokenizer
from paddlenlp.transformers import create_bigbird_rand_mask_idx_list
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.utils.log import logger

import args

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

MODEL_CLASSES = {
    "bigbird": (BigBirdForSequenceClassification, BigBirdTokenizer),
}


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    input_ids = [tokenizer.cls_id]
    token_type_ids = None

    if (int(is_test) + len(example)) == 2:
        input_ids.extend(
            tokenizer.convert_tokens_to_ids(
                tokenizer(example['sentence'])[:max_seq_length - 2]))
        input_ids.append(tokenizer.sep_id)
        input_len = len(input_ids)
        token_type_ids = input_len * [0]
    else:
        input_ids1 = tokenizer.convert_tokens_to_ids(
            tokenizer(example['sentence1']))
        input_ids2 = tokenizer.convert_tokens_to_ids(
            tokenizer(example['sentence2']))
        total_len = len(input_ids1) + len(
            input_ids2) + tokenizer.num_special_tokens_to_add(pair=True)
        if total_len > max_seq_length:
            input_ids1, input_ids2, _ = tokenizer.truncate_sequences(
                input_ids1, input_ids2, total_len - max_seq_length)
        input_ids.extend(input_ids1)
        input_ids.append(tokenizer.sep_id)
        input_len1 = len(input_ids)

        input_ids.extend(input_ids2)
        input_ids.append(tokenizer.sep_id)
        input_len2 = len(input_ids) - input_len1

        token_type_ids = input_len1 * [0] + input_len2 * [1]

    input_len = len(input_ids)
    if input_len < max_seq_length:
        input_ids.extend([tokenizer.pad_id] * (max_seq_length - input_len))
        token_type_ids.extend([tokenizer.pad_token_type_id] *
                              (max_seq_length - input_len))

    if not is_test:
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def collect_data(samples, dataset, config):
    stack_fn = Stack(dtype="int64" if dataset.label_list else "float32")
    stack_fn1 = Stack()

    num_fields = len(samples[0])
    out = [None] * num_fields
    out[0] = stack_fn1([x[0] for x in samples])  # input_ids
    out[1] = stack_fn1([x[1] for x in samples])  # token_type_ids
    if num_fields >= 2:
        out[2] = stack_fn(x[2] for x in samples)  # labels
    seq_len = len(out[0][0])
    # Construct the random attention mask for the random attention
    rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
        config["num_layers"], seq_len, seq_len, config["nhead"],
        config["block_size"], config["window_size"],
        config["num_global_blocks"], config["num_rand_blocks"], config["seed"])
    out.extend(rand_mask_idx_list)
    return out


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch[:3]
        rand_mask_idx_list = batch[3:]
        # run forward
        logits = model(input_ids,
                       segment_ids,
                       rand_mask_idx_list=rand_mask_idx_list)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        logger.info(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4],
            ))
    elif isinstance(metric, Mcc):
        logger.info("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]))
    elif isinstance(metric, PearsonAndSpearman):
        logger.info(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]))
    else:
        logger.info("eval loss: %f, acc: %s, " % (loss.numpy(), res))
    model.train()


def do_train(args):
    paddle.set_device(args.device)
    worker_num = paddle.distributed.get_world_size()
    if worker_num > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = load_dataset('glue', args.task_name, splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    # In finetune task, bigbird performs better when setting dropout to zero.
    model = model_class.from_pretrained(args.model_name_or_path,
                                        num_classes=num_classes,
                                        attn_dropout=0.0,
                                        hidden_dropout_prob=0.0)
    if worker_num > 1:
        model = paddle.DataParallel(model)
    config = getattr(model, model_class.base_model_prefix).config

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_list=train_ds.label_list,
                         max_seq_length=args.max_encoder_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = partial(collect_data, dataset=train_ds, config=config)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)

    if args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            'glue', args.task_name, splits=["dev_matched", "dev_mismatched"])

        dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_ds_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_ds_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_ds = load_dataset('glue', args.task_name, splits='dev')
        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
        dev_data_loader = DataLoader(dataset=dev_ds,
                                     batch_sampler=dev_batch_sampler,
                                     collate_fn=batchify_fn,
                                     num_workers=0,
                                     return_list=True)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    global_step = 0
    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch[:3]
            rand_mask_idx_list = batch[3:]
            # run forward
            logits = model(input_ids,
                           segment_ids,
                           rand_mask_idx_list=rand_mask_idx_list)
            loss = loss_fct(logits, labels)
            # run backward and update params
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric,
                             dev_data_loader_mismatched)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(
                        args.output_dir, "%s_ft_model_%d.pdparams" %
                        (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
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
    args = args.parse_args()
    print_arguments(args)
    assert args.device in [
        "cpu", "gpu", "xpu"
    ], "Invalid device! Available device should be cpu, gpu, or xpu."
    do_train(args)
