# coding: utf-8
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
import time
import random
import argparse
import numpy as np

import paddle
import paddle.nn as nn

from datasets import load_dataset

from paddlenlp.data import Stack, Dict, Pad, Tuple
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ErnieForMultipleChoice, ErnieTokenizer
from paddlenlp.transformers import RobertaForMultipleChoice, RobertaTokenizer

MODEL_CLASSES = {
    "ernie": (ErnieForMultipleChoice, ErnieTokenizer),
    "roberta": (RobertaForMultipleChoice, RobertaTokenizer)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
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
        default="best_c3_model",
        type=str,
        help="The  path of the checkpoints .", )
    parser.add_argument(
        "--num_train_epochs",
        default=8,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--weight_decay",
        default=0.01,
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
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
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
def evaluate(model, loss_fct, dev_data_loader, metric):
    metric.reset()
    criterion = paddle.nn.loss.CrossEntropyLoss()
    model.eval()
    for step, batch in enumerate(dev_data_loader):
        input_ids, segment_ids, label_id = batch
        logits = model(input_ids=input_ids, token_type_ids=segment_ids)
        loss = loss_fct(logits, label_id)
        correct = metric.compute(logits, label_id)
        metric.update(correct)
    acc = metric.accumulate()
    model.train()
    return acc


def do_train(args):
    max_seq_length = args.max_seq_length
    max_num_choices = 4

    def preprocess_function(examples):
        def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
            """Truncates a sequence tuple in place to the maximum length."""
            # This is a simple heuristic which will always truncate the longer
            # sequence one token at a time. This makes more sense than
            # truncating an equal percent of tokens from each, since if one
            # sequence is very short then each token that's truncated likely
            # contains more information than a longer sequence.
            while True:
                total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
                if total_length <= max_length:
                    break
                if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(
                        tokens_c):
                    tokens_a.pop()
                elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(
                        tokens_c):
                    tokens_b.pop()
                else:
                    tokens_c.pop()

        num_examples = len(examples.data["question"])
        result = {"input_ids": [], "token_type_ids": [], "labels": []}
        for idx in range(num_examples):
            text = '\n'.join(examples.data["context"][idx]).lower()
            question = examples.data["question"][idx].lower()
            choice_list = examples.data["choice"][idx]
            choice_list = [choice.lower() for choice in choice_list]
            answer = examples.data["answer"][idx].lower()
            label = choice_list.index(answer)

            tokens_t = tokenizer.tokenize(text)
            tokens_q = tokenizer.tokenize(question)

            tokens_t_list = []
            tokens_c_list = []

            # Pad each new example for axis=1, [batch_size, num_choices, seq_len]
            while len(choice_list) < max_num_choices:
                choice_list.append('无效答案')

            for choice in choice_list:
                tokens_c = tokenizer.tokenize(choice.lower())
                _truncate_seq_tuple(tokens_t, tokens_q, tokens_c,
                                    max_seq_length - 4)

                tokens_c = tokens_q + ["[SEP]"] + tokens_c
                tokens_t_list.append(tokens_t)
                tokens_c_list.append(tokens_c)

            new_data = tokenizer(
                tokens_t_list,
                text_pair=tokens_c_list,
                is_split_into_words=True)

            # Pad each new example for axis=2 of [batch_size, num_choices, seq_len],
            # because length of each choice could be different.
            input_ids = Pad(
                axis=0, pad_val=tokenizer.pad_token_id)(new_data["input_ids"])
            token_type_ids = Pad(
                axis=0,
                pad_val=tokenizer.pad_token_id)(new_data["token_type_ids"])

            # Final shape of input_ids: [batch_size, num_choices, seq_len]
            result["input_ids"].append(input_ids)
            result["token_type_ids"].append(token_type_ids)
            result["labels"].append([label])
            if (idx + 1) % 1000 == 0:
                print(idx + 1, "samples has been processed.")
        return result

    paddle.set_device(args.device)
    set_seed(args)

    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path, num_choices=max_num_choices)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    train_ds = load_dataset("clue", "c3", split="train")
    column_names = train_ds.column_names
    train_ds = train_ds.map(preprocess_function,
                            batched=True,
                            batch_size=len(train_ds),
                            num_proc=8,
                            remove_columns=column_names)
    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=1, pad_val=tokenizer.pad_token_id),  # input
        'token_type_ids': Pad(axis=1, pad_val=tokenizer.pad_token_type_id),  # segment
        'labels': Stack(dtype="int64")  # label
    }): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = load_dataset("clue", "c3", split="validation")
    dev_ds = dev_ds.map(preprocess_function,
                        batched=True,
                        batch_size=len(dev_ds),
                        remove_columns=column_names,
                        num_proc=8)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

    num_training_steps = len(train_data_loader) * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=grad_clip)
    loss_fct = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.train()
    global_step = 0
    best_acc = 0.0
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, label = batch
            logits = model(input_ids=input_ids, token_type_ids=segment_ids)
            loss = loss_fct(logits, label)
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
                acc = evaluate(model, loss_fct, dev_data_loader, metric)
                print("eval acc: %.5f, eval done total : %s s" %
                      (acc, time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0 and acc > best_acc:
                    best_acc = acc
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
            if global_step >= num_training_steps:
                print("best_acc: ", best_acc)
                return
    print("best_acc: ", best_acc)


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
