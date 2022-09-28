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
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.transformers import TinyBertForSequenceClassification, TinyBertTokenizer
from paddlenlp.transformers.distill_utils import to_distill

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

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
    "bert": (BertForSequenceClassification, BertTokenizer),
    "tinybert": (TinyBertForSequenceClassification, TinyBertTokenizer),
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
    parser.add_argument(
        "--model_type",
        default="tinybert",
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--teacher_model_type",
        default="bert",
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--student_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])),
    )
    parser.add_argument("--teacher_path",
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
        "--glue_dir",
        default="/root/.paddlenlp/datasets/Glue/",
        type=str,
        required=False,
        help="The Glue directory.",
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
    parser.add_argument(
        "--T",
        default=1,
        type=int,
        help="Temperature for softmax",
    )
    parser.add_argument(
        "--use_aug",
        action="store_true",
        help="Whether to use augmentation data to train.",
    )
    parser.add_argument(
        "--intermediate_distill",
        action="store_true",
        help=
        "Whether distilling intermediate layers. If False, it means prediction layer distillation.",
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
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
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
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print("acc: %s, precision: %s, recall: %s, f1: %s, acc and f1: %s, " % (
            res[0],
            res[1],
            res[2],
            res[3],
            res[4],
        ),
              end='')
    elif isinstance(metric, Mcc):
        print("mcc: %s, " % (res[0]), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print("pearson: %s, spearman: %s, pearson and spearman: %s, " %
              (res[0], res[1], res[2]),
              end='')
    else:
        print("acc: %s, " % (res), end='')
    model.train()
    return res[0] if isinstance(metric, (AccuracyAndF1, Mcc,
                                         PearsonAndSpearman)) else res


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
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    else:
        example = tokenizer(example['sentence1'],
                            text_pair=example['sentence2'],
                            max_seq_len=max_seq_length)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.use_aug:
        aug_data_file = os.path.join(
            os.path.join(args.glue_dir, args.task_name), "train_aug.tsv"),
        train_ds = load_dataset('glue',
                                args.task_name,
                                data_files=aug_data_file)
    else:
        train_ds = load_dataset('glue', args.task_name, splits='train')
    tokenizer = tokenizer_class.from_pretrained(args.student_model_name_or_path)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         label_list=train_ds.label_list,
                         max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
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

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    student = model_class.from_pretrained(args.student_model_name_or_path,
                                          num_classes=num_classes)
    teacher_model_class, _ = MODEL_CLASSES[args.teacher_model_type]
    teacher = teacher_model_class.from_pretrained(args.teacher_path,
                                                  num_classes=num_classes)

    if paddle.distributed.get_world_size() > 1:
        student = paddle.DataParallel(student, find_unused_parameters=True)
        teacher = paddle.DataParallel(teacher, find_unused_parameters=True)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in student.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=student.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    ce_loss_fct = paddle.nn.CrossEntropyLoss(soft_label=True)
    mse_loss_fct = paddle.nn.MSELoss()

    metric = metric_class()

    teacher = to_distill(teacher,
                         return_attentions=True,
                         return_qkv=False,
                         return_layer_outputs=True)
    student = to_distill(student,
                         return_attentions=True,
                         return_qkv=False,
                         return_layer_outputs=True)
    pad_token_id = 0
    global_step = 0
    tic_train = time.time()
    best_res = 0.0

    def cal_intermediate_distill_loss(student, teacher):
        loss_hidden, loss_attn = 0, 0
        # Calculate emb loss(hidden_states[0]) and hidden states loss.
        for i in range(len(student.outputs.hidden_states)):
            # While using tinybert-4l-312d, tinybert-6l-768d, tinybert-4l-312d-zh, tinybert-6l-768d-zh
            # student_hidden = student.tinybert.fit_dense(student.outputs.hidden_states[i])
            # While using tinybert-4l-312d-v2, tinybert-6l-768d-v2
            if isinstance(student, paddle.DataParallel):
                student_hidden = student._layers.tinybert.fit_denses[i](
                    student.outputs.hidden_states[i])
            else:
                student_hidden = student.tinybert.fit_denses[i](
                    student.outputs.hidden_states[i])
            loss_hidden += mse_loss_fct(student_hidden,
                                        teacher.outputs.hidden_states[2 * i])
        for i in range(len(student.outputs.attentions)):
            attn_student = student.outputs.attentions[i]
            attn_teacher = teacher.outputs.attentions[2 * i + 1]
            loss_attn += mse_loss_fct(attn_student, attn_teacher)
        loss = loss_hidden + loss_attn
        return loss

    distill_part = "intermediate" if args.intermediate_distill else "pred"

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch
            logits = student(input_ids, segment_ids)
            with paddle.no_grad():
                teacher_logits = teacher(input_ids, segment_ids)

            if args.intermediate_distill:
                loss = cal_intermediate_distill_loss(student, teacher)
            else:
                loss = ce_loss_fct(logits / args.T,
                                   F.softmax(teacher_logits / args.T))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
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
                    res = evaluate(student, metric, dev_data_loader_matched)
                    evaluate(student, metric, dev_data_loader_mismatched)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    res = evaluate(student, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if (best_res < res and global_step < num_training_steps
                        or global_step == num_training_steps
                    ) and paddle.distributed.get_rank() == 0:
                    if global_step < num_training_steps:
                        output_dir = os.path.join(
                            args.output_dir, "%s_distill_model_%d.pdparams" %
                            (distill_part, global_step))
                    else:
                        output_dir = os.path.join(
                            args.output_dir,
                            "%s_distill_model_final.pdparams" % (distill_part))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    model_to_save = student._layers if isinstance(
                        student, paddle.DataParallel) else student
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    best_res = res

            if global_step >= num_training_steps:
                return


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
