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
import os
import sys
import random
import time
import math
import json
import distutils.util
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
from paddle.metric import Accuracy

from paddlenlp.datasets import load_dataset
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model or shortcut name.")
    parser.add_argument(
        "--output_dir",
        default="best_clue_model",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--logging_steps",
                        type=int,
                        default=100,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help=
        "Linear warmup over warmup_steps. If > 0: Override warmup_proportion")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Linear warmup proportion over total steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumualte before performing a backward/update pass."
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether do train.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether do train.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether do predict.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--save_best_model",
        default=True,
        type=distutils.util.strtobool,
        help="Whether to save best model.",
    )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="The max value of grad norm.")
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
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    logger.info("eval loss: %f, acc: %s, " % (loss.numpy(), res))
    model.train()
    return res


def convert_example(example,
                    tokenizer,
                    label_list,
                    is_test=False,
                    max_seq_length=512):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = np.array(example["label"], dtype="int64")
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {'sentence1': sentence1, 'sentence2': example['abst']}
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(
            pronoun_idx +
            len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx +
                               len(query))] == query, "query: {}".format(query)
        if pronoun_idx > query_idx:
            text_list.insert(query_idx, "_")
            text_list.insert(query_idx + len(query) + 1, "_")
            text_list.insert(pronoun_idx + 2, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_list.insert(pronoun_idx, "[")
            text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_list.insert(query_idx + 2, "_")
            text_list.insert(query_idx + len(query) + 2 + 1, "_")
        text = "".join(text_list)
        example['sentence'] = text
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)
    if not is_test:
        example["labels"] = label
    return example


def do_eval(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]

    dev_ds = load_dataset('clue', args.task_name, splits='dev')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    trans_func = partial(convert_example,
                         label_list=dev_ds.label_list,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    batchify_fn = DataCollatorWithPadding(tokenizer)

    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=batchify_fn,
                                 num_workers=0,
                                 return_list=True)

    num_classes = 1 if dev_ds.label_list == None else len(dev_ds.label_list)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    metric = metric_class()
    best_acc = 0.0
    global_step = 0
    tic_train = time.time()
    model.eval()
    metric.reset()
    for batch in dev_data_loader:
        labels = batch.pop("labels")
        logits = model(**batch)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    logger.info("acc: %s\n, " % (res))


def do_train(args):
    assert args.batch_size % args.gradient_accumulation_steps == 0, \
        "Please make sure argmument `batch_size` must be divisible by `gradient_accumulation_steps`."
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    train_ds, dev_ds = load_dataset('clue',
                                    args.task_name,
                                    splits=('train', 'dev'))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example,
                         label_list=train_ds.label_list,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    train_ds = train_ds.map(trans_func, lazy=True)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    batchify_fn = DataCollatorWithPadding(tokenizer)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=batchify_fn,
                                   num_workers=0,
                                   return_list=True)
    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=batchify_fn,
                                 num_workers=0,
                                 return_list=True)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    if args.dropout != 0.1:
        update_model_dropout(model, args.dropout)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps / args.gradient_accumulation_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(
            train_data_loader
        ) * args.num_train_epochs / args.gradient_accumulation_steps
        num_train_epochs = args.num_train_epochs

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
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))

    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    best_acc = 0.0
    global_step = 0
    tic_train = time.time()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            labels = batch.pop("labels")
            logits = model(**batch)
            loss = loss_fct(logits, labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                        % (global_step, num_training_steps, epoch, step,
                           paddle.distributed.get_rank(), loss,
                           optimizer.get_lr(), args.logging_steps /
                           (time.time() - tic_train)))
                    tic_train = time.time()
                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    tic_eval = time.time()
                    acc = evaluate(model, loss_fct, metric, dev_data_loader)
                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                    if acc > best_acc:
                        best_acc = acc
                        if args.save_best_model:
                            output_dir = args.output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # Need better way to get inner model of DataParallel
                            model_to_save = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                if global_step >= num_training_steps:
                    logger.info("best_result: %.2f" % (best_acc * 100))
                    return
    logger.info("best_result: %.2f" % (best_acc * 100))


def do_predict(args):
    paddle.set_device(args.device)
    args.task_name = args.task_name.lower()

    train_ds, test_ds = load_dataset('clue',
                                     args.task_name,
                                     splits=('train', 'test'))
    if args.task_name == "cluewsc2020" or args.task_name == "tnews":
        test_ds_10 = load_dataset('clue', args.task_name, splits="test1.0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_list=train_ds.label_list,
                         max_seq_length=args.max_seq_length,
                         is_test=True)

    batchify_fn = DataCollatorWithPadding(tokenizer)

    test_ds = test_ds.map(trans_func, lazy=True)
    test_batch_sampler = paddle.io.BatchSampler(test_ds,
                                                batch_size=args.batch_size,
                                                shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_sampler=test_batch_sampler,
                                  collate_fn=batchify_fn,
                                  num_workers=0,
                                  return_list=True)
    if args.task_name == "cluewsc2020" or args.task_name == "tnews":
        test_ds_10 = test_ds_10.map(trans_func, lazy=True)
        test_batch_sampler_10 = paddle.io.BatchSampler(
            test_ds_10, batch_size=args.batch_size, shuffle=False)
        test_data_loader_10 = DataLoader(dataset=test_ds_10,
                                         batch_sampler=test_batch_sampler_10,
                                         collate_fn=batchify_fn,
                                         num_workers=0,
                                         return_list=True)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    prediction_filename = args.task_name

    if args.task_name == 'ocnli':
        prediction_filename = 'ocnli_50k'
    elif args.task_name == "cluewsc2020":
        prediction_filename = "cluewsc" + "11"
    elif args.task_name == "tnews":
        prediction_filename = args.task_name + "11"

    # For version 1.1
    f = open(
        os.path.join(args.output_dir, prediction_filename + "_predict.json"),
        'w')
    preds = []
    for step, batch in enumerate(test_data_loader):
        with paddle.no_grad():
            logits = model(**batch)
        pred = paddle.argmax(logits, axis=1).numpy().tolist()
        preds += pred
    for idx, pred in enumerate(preds):
        j = json.dumps({"id": idx, "label": train_ds.label_list[pred]})
        f.write(j + "\n")

    # For version 1.0
    if args.task_name == "cluewsc2020" or args.task_name == "tnews":
        prediction_filename = args.task_name + "10"
        if args.task_name == "cluewsc2020":
            prediction_filename = "cluewsc10"
        f = open(
            os.path.join(args.output_dir,
                         prediction_filename + "_predict.json"), 'w')

        preds = []
        for step, batch in enumerate(test_data_loader_10):
            with paddle.no_grad():
                logits = model(**batch)
            pred = paddle.argmax(logits, axis=1).numpy().tolist()
            preds += pred
        for idx, pred in enumerate(preds):
            j = json.dumps({"id": idx, "label": train_ds.label_list[pred]})
            f.write(j + "\n")


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def update_model_dropout(model, p=0.0):
    model.base_model.embeddings.dropout.p = p
    for i in range(len(model.base_model.encoder.layers)):
        model.base_model.encoder.layers[i].dropout.p = p
        model.base_model.encoder.layers[i].dropout1.p = p
        model.base_model.encoder.layers[i].dropout2.p = p


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.do_train:
        do_train(args)
    if args.do_eval:
        do_eval(args)
    if args.do_predict:
        do_predict(args)
