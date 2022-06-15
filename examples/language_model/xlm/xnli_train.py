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

import os
import argparse
import random
import math
import time
import distutils.util
from functools import partial
import numpy as np

import paddle
import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
from paddlenlp.transformers import XLMForSequenceClassification, XLMTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddle.metric import Accuracy
from paddle.optimizer import Adam

all_languages = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--learning_rate",
                        default=2e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Dropout rate.")
    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=200,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=24544,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU/XPU for training.",
    )
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument("--use_amp",
                        type=distutils.util.strtobool,
                        default=False,
                        help="Enable mixed precision training.")
    parser.add_argument("--scale_loss",
                        type=float,
                        default=2**15,
                        help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, metric, data_loader, language, tokenizer):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, attention_mask, labels = batch
        # add lang_ids
        lang_ids = paddle.ones_like(input_ids) * tokenizer.lang2id[language]
        logits = model(input_ids, langs=lang_ids, attention_mask=attention_mask)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    print("[%s] acc: %s " % (language.upper(), res))
    model.train()
    return res


def convert_example(example, tokenizer, max_seq_length=256, language="en"):
    """convert a example into necessary features"""
    # Get the label
    label = example["label"]
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    # Convert raw text to feature
    example = tokenizer(premise,
                        text_pair=hypothesis,
                        max_length=max_seq_length,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                        lang=language)
    return example["input_ids"], example["attention_mask"], label


def get_test_dataloader(args, language, batchify_fn, tokenizer):
    # make sure language is `language``
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         language=language)
    test_ds = load_dataset("xnli", language, splits="test")
    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = BatchSampler(test_ds,
                                      batch_size=args.batch_size * 4,
                                      shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  return_list=True)
    return test_data_loader


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    tokenizer = XLMTokenizer.from_pretrained(args.model_name_or_path)

    # define train dataset language
    language = "en"
    train_ds = load_dataset("xnli", language, splits="train")
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         language=language)

    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = DistributedBatchSampler(train_ds,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=0, dtype="int64"),  # attention_mask
        Stack(dtype="int64")  # labels
    ): fn(samples)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)

    model = XLMForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=3, dropout=args.dropout)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    optimizer = Adam(learning_rate=args.learning_rate,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=args.adam_epsilon,
                     parameters=model.parameters())

    loss_fct = nn.CrossEntropyLoss()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)
    metric = Accuracy()

    global_step = 0
    tic_train = time.time()
    max_test_acc = 0.0
    print(f"num_training_steps {num_training_steps}")
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, attention_mask, labels = batch
            lang_ids = paddle.ones_like(input_ids) * tokenizer.lang2id[language]

            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(input_ids,
                               langs=lang_ids,
                               attention_mask=attention_mask)
                loss = loss_fct(logits, labels)

            if args.use_amp:
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.minimize(optimizer, scaled_loss)
            else:
                loss.backward()
                optimizer.step()

            optimizer.clear_grad()
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                all_languages_acc = []
                for language in all_languages:
                    test_data_loader = get_test_dataloader(
                        args, language, batchify_fn, tokenizer)
                    acc = evaluate(model, metric, test_data_loader, language,
                                   tokenizer)
                    all_languages_acc.append(acc)
                test_mean_acc = np.mean(all_languages_acc)
                print("test mean acc: %.4f" % test_mean_acc)

                if paddle.distributed.get_rank() == 0:
                    if test_mean_acc >= max_test_acc:
                        max_test_acc = test_mean_acc
                        output_dir = os.path.join(args.output_dir, "best_model")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Need better way to get inner model of DataParallel
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        print("best test mean acc: %.4f" % max_test_acc)
                        print("Save model and tokenizer to %s" % output_dir)

            if global_step >= num_training_steps:
                return


def print_arguments(args):
    """print arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
