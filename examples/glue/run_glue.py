# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import ElectraForSequenceClassification, ElectraTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

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

MODEL_CLASSES = {
    "bert": (BertForSequenceClassification, BertTokenizer),
    "electra": (ElectraForSequenceClassification, ElectraTokenizer),
    "ernie": (ErnieForSequenceClassification, ErnieTokenizer),
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
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--n_gpu",
        default=1,
        type=int,
        help="number of gpus to use, 0 for cpu.")
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


def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: %f, acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, "
            % (
                loss.numpy(),
                res[0],
                res[1],
                res[2],
                res[3],
                res[4], ),
            end='')
    elif isinstance(metric, Mcc):
        print("eval loss: %f, mcc: %s, " % (loss.numpy(), res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]),
            end='')
    else:
        print("eval loss: %f, acc: %s, " % (loss.numpy(), res), end='')
    model.train()


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    """convert a glue example into necessary features"""

    def _truncate_seqs(seqs, max_seq_length):
        if len(seqs) == 1:  # single sentence
            # Account for [CLS] and [SEP] with "- 2"
            seqs[0] = seqs[0][0:(max_seq_length - 2)]
        else:  # Sentence pair
            # Account for [CLS], [SEP], [SEP] with "- 3"
            tokens_a, tokens_b = seqs
            max_seq_length -= 3
            while True:  # Truncate with longest_first strategy
                total_length = len(tokens_a) + len(tokens_b)
                if total_length <= max_seq_length:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        return seqs

    def _concat_seqs(seqs, separators, seq_mask=0, separator_mask=1):
        concat = sum((seq + sep for sep, seq in zip(separators, seqs)), [])
        segment_ids = sum(
            ([i] * (len(seq) + len(sep))
             for i, (sep, seq) in enumerate(zip(separators, seqs))), [])
        if isinstance(seq_mask, int):
            seq_mask = [[seq_mask] * len(seq) for seq in seqs]
        if isinstance(separator_mask, int):
            separator_mask = [[separator_mask] * len(sep) for sep in separators]
        p_mask = sum((s_mask + mask
                      for sep, seq, s_mask, mask in zip(
                          separators, seqs, seq_mask, separator_mask)), [])
        return concat, segment_ids, p_mask

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
        example = tokenizer(example[0], max_seq_len=max_seq_length)
    else:
        example = tokenizer(
            example[0], text_pair=example[1], max_seq_len=max_seq_length)
    '''
    tokens_raw = [tokenizer(l) for l in example]
    # Truncate to the truncate_length,
    tokens_trun = _truncate_seqs(tokens_raw, max_seq_length)
    # Concate the sequences with special tokens
    tokens_trun[0] = [tokenizer.cls_token] + tokens_trun[0]
    tokens, segment_ids, _ = _concat_seqs(tokens_trun, [[tokenizer.sep_token]] *
                                          len(tokens_trun))
    # Convert the token to ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    valid_length = len(input_ids)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    # input_mask = [1] * len(input_ids)
    '''
    if not is_test:
        return example['input_ids'], example['segment_ids'], len(example[
            'input_ids']), label
    else:
        return example['input_ids'], example['segment_ids'], len(example[
            'input_ids'])


def do_train(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    dataset_class, metric_class = TASK_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_dataset = dataset_class.get_datasets(["train"])
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_dataset.get_labels(),
        max_seq_length=args.max_seq_length)
    train_dataset = train_dataset.apply(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(),  # length
        Stack(dtype="int64" if train_dataset.get_labels() else "float32")  # label
    ): [data for i, data in enumerate(fn(samples)) if i != 2]
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
            dev_dataset, batch_size=args.batch_size, shuffle=False)
        dev_data_loader = DataLoader(
            dataset=dev_dataset,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_classes = 1 if train_dataset.get_labels() == None else len(
        train_dataset.get_labels())
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ])

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if train_dataset.get_labels(
    ) else paddle.nn.loss.MSELoss()

    metric = metric_class()

    global_step = 0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch
            logits = model(input_ids, segment_ids)
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
            if global_step % args.save_steps == 0:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    evaluate(model, loss_fct, metric,
                             dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if (not args.n_gpu > 1) or paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir,
                                              "%s_ft_model_%d.pdparams" %
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
    args = parse_args()
    print_arguments(args)
    if args.n_gpu > 1:
        paddle.distributed.spawn(do_train, args=(args, ), nprocs=args.n_gpu)
    else:
        do_train(args)
