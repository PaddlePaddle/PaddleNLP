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
import os
import random
import time
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Accuracy

from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.xlnet.modeling import XLNetPretrainedModel, XLNetForSequenceClassification
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

final_res = "Not evaluated yet!"

TASK_CLASSES = {
    "cola": (GlueCoLA, Mcc),
    "sst-2": (GlueSST2, Accuracy),
    "mrpc": (GlueMRPC, AccuracyAndF1),
    "sts-b": (GlueSTSB, PearsonAndSpearman),
    "qqp": (GlueQQP, AccuracyAndF1),
    "mnli": (GlueMNLI, Accuracy),
    "qnli": (GlueQNLI, Accuracy),
    "rte": (GlueRTE, Accuracy),
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
        ", ".join(TASK_CLASSES.keys()), )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(XLNetPretrainedModel.pretrained_init_configuration.keys()), )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",)

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")

    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--eager_run", type=eval, default=True, help="Use dygraph mode.")

    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="number of gpus to use, 0 for cpu.")

    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")

    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    global final_res
    for batch in data_loader:
        input_ids, token_type_ids, attention_mask, labels = batch
        logits = model(input_ids, token_type_ids, attention_mask)[0]
        loss = loss_fct(logits, labels)
        losses.append(loss.detach().numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s"
            % (np.average(losses), res[0], res[1], res[2], res[3], res[4]))

        final_res = "final:    acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s"\
                    % (res[0], res[1], res[2], res[3], res[4])
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s" % (np.average(losses), res[0]))
        final_res = "final:    mcc: %s" % (res[0])
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s"
            % (np.average(losses), res[0], res[1], res[2]))
        final_res = "final:    pearson: %s, spearman: %s, pearson and spearman: %s" % (res[0], res[1], res[2])
    else:
        print("eval loss: %f, acc: %s" % (np.average(losses), res))
        final_res = "final:    acc: %s" % res
    model.train()


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
        label = example[-1]
        example = example[:-1]
        # Create label maps if classification task
        if label_list:
            label_map = {}
            for (i, l) in enumerate(label_list):
                label_map[l] = i
            label = label_map[label]
        label = np.array([label], dtype=label_dtype)

    # Tokenize raw text
    if len(example) == 1:
        example = tokenizer(example[0], max_seq_len=max_seq_length, return_attention_mask=True)
    else:
        example = tokenizer(example[0], text_pair=example[1], max_seq_len=max_seq_length, return_attention_mask=True)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], example['attention_mask'], \
               len(example['input_ids']), label
    else:
        return example['input_ids'], example['token_type_ids'], example['attention_mask'], \
               len(example['input_ids'])


def do_train(args):
    paddle.enable_static() if not args.eager_run else None
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    model_class, tokenizer_class = XLNetForSequenceClassification, XLNetTokenizer

    train_dataset = dataset_class.get_datasets(["train"])

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    label_list = train_dataset.get_labels()
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=label_list,
        max_seq_length=args.max_seq_length)

    train_dataset = train_dataset.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.sp_model.PieceToId(tokenizer.pad_token), pad_right=False),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, pad_right=False),                        # token_type
        Pad(axis=0, pad_val=0, pad_right=False),                                                  # attention_mask
        Stack(),                                                                                  # length
        Stack(dtype="int64" if label_list else "float32"),                                        # label
    ): [data for i, data in enumerate(fn(samples)) if i != 3]

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    if args.task_name == "mnli":
        dev_dataset_matched, dev_dataset_mismatched = dataset_class.get_datasets(
            ["dev_matched", "dev_mismatched"])
        dev_dataset_matched = dev_dataset_matched.apply(trans_func, lazy=True)
        dev_dataset_mismatched = dev_dataset_mismatched.apply(
            trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.BatchSampler(
            dev_dataset_matched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_matched = DataLoader(
            dataset=dev_dataset_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.BatchSampler(
            dev_dataset_mismatched, batch_size=args.batch_size, shuffle=False)
        dev_data_loader_mismatched = DataLoader(
            dataset=dev_dataset_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_dataset = dataset_class.get_datasets(["dev"])
        dev_dataset = dev_dataset.apply(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.BatchSampler(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False)

        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_classes = 1 if train_dataset.get_labels() is None else len(train_dataset.get_labels())
    model = XLNetForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm", "LayerNorm"])
        ])

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()

    global_step = 0
    tic_train = time.time()
    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, attention_mask, labels = batch
            logits = model(input_ids, token_type_ids, attention_mask, labels=labels)[0]
            loss = loss_fct(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric, dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_ft_model_%d" %
                                              (args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                if global_step == num_training_steps:
                    print(final_res)
                tic_train += time.time() - tic_eval


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
