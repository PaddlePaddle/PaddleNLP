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
import random
import time
import math
from functools import partial

import numpy as np
import paddle

import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.utils.log import logger

from trainer_base import TrainerBase


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader, label_num, mode="valid"):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    logger.info("%s: eval loss: %f, precision: %f, recall: %f, f1: %f" %
                (mode, avg_loss, precision, recall, f1_score))
    model.train()

    return f1_score


def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    return tokenized_input


class NerTrainer(TrainerBase):
    def __init__(self, train_ds, dev_ds, model, tokenizer, args, *arg,
                 **kwargs):
        super().__init__()
        self.rank = paddle.distributed.get_rank()
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        if "test_ds" in kwargs.keys():
            self.test_ds = kwargs["test_ds"]
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.dataloader_inner()
        self.prepare_train_config()
        self.print_config()

    def dataloader_inner(self):
        label_list = self.train_ds.label_list
        label_num = len(label_list)
        no_entity_id = label_num - 1

        trans_fn = partial(
            tokenize_and_align_labels,
            tokenizer=self.tokenizer,
            no_entity_id=no_entity_id,
            max_seq_len=self.args.max_seq_length)

        ignore_label = -100

        batchify_fn = lambda samples, fn=Dict({
            'input_ids': Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype='int32'),  # input
            'token_type_ids': Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype='int32'),  # segment
            'seq_len': Stack(dtype='int64'),  # seq_len
            'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
        }): fn(samples)

        self.train_dl = self.create_dataloader(
            self.train_ds, "train", self.args.batch_size, batchify_fn, trans_fn)
        self.dev_dl = self.create_dataloader(
            self.dev_ds, "dev", self.args.batch_size, batchify_fn, trans_fn)
        self.test_dl = self.create_dataloader(
            self.test_ds, "test", self.args.batch_size, batchify_fn, trans_fn)

    def train(self):
        ignore_label = -100
        label_num = len(self.train_ds.label_list)

        loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
        metric = ChunkEvaluator(label_list=self.args.label_list)

        global_step = 0
        tic_train = time.time()
        best_dev_f1 = -1
        corr_test_f1 = -1

        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.train_dl):
                global_step += 1
                input_ids, token_type_ids, _, labels = batch
                logits = self.model(input_ids, token_type_ids)
                loss = loss_fct(logits, labels)
                avg_loss = paddle.mean(loss)

                if global_step % self.args.logging_steps == 0:
                    logger.info(
                        "global step %d/%d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, self.args.num_training_steps, epoch,
                           step, avg_loss,
                           self.args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()

                avg_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                if global_step % self.args.valid_steps == 0 or global_step == self.args.num_training_steps:
                    if paddle.distributed.get_rank() == 0:
                        dev_f1 = evaluate(self.model, loss_fct, metric,
                                          self.dev_dl, label_num, "valid")
                        test_f1 = evaluate(self.model, loss_fct, metric,
                                           self.test_dl, label_num, "test")
                        if dev_f1 > best_dev_f1:
                            best_dev_f1 = dev_f1
                            corr_test_f1 = test_f1
                        logger.warning(
                            "Currently, best_dev_f1: %.4f, corr_test_f1: %.4f" %
                            (best_dev_f1, corr_test_f1))

                if global_step >= self.args.num_training_steps:
                    logger.warning(
                        "Currently, best_dev_f1: %.4f, corr_test_f1: %.4f" %
                        (best_dev_f1, corr_test_f1))
                    return
