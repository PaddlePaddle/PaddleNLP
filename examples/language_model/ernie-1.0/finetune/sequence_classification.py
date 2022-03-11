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

import paddle
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
import numpy as np

from paddlenlp.data import Stack, Tuple, Pad, Dict

import argparse
import os
import sys
import random
import time
import math
import copy
import yaml
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import AutoModelForTokenClassification
from paddlenlp.transformers import AutoModelForQuestionAnswering
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger


class BaseTrainer(object):
    def create_dataloader(self,
                          dataset,
                          mode='train',
                          batch_size=16,
                          batchify_fn=None,
                          trans_fn=None,
                          batched=False):
        if trans_fn:
            dataset = dataset.map(trans_fn, batched=batched)

        shuffle = True if mode == 'train' else False
        if mode == 'train':
            batch_sampler = paddle.io.DistributedBatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)
        else:
            batch_sampler = paddle.io.BatchSampler(
                dataset, batch_size=batch_size, shuffle=shuffle)

        return paddle.io.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    def prepare_train_config(self):
        if self.args.max_steps > 0:
            self.args.num_training_steps = self.args.max_steps
            self.args.num_train_epochs = math.ceil(
                self.args.num_training_steps / len(self.train_dl))

        else:
            self.args.num_training_steps = len(
                self.train_dl) * self.args.num_train_epochs
            self.args.num_train_epochs = self.args.num_train_epochs

        if self.args.num_training_steps // self.args.valid_steps < 20:
            exp_step = self.args.num_training_steps / 20
            exp_step = max(int(exp_step - exp_step % 10), 10)
            logger.info("Set eval step to %d" % exp_step)
            self.args.valid_steps = exp_step

        warmup = self.args.warmup_steps if self.args.warmup_steps > 0 else self.args.warmup_proportion

        self.lr_scheduler = LinearDecayWithWarmup(
            self.args.learning_rate, self.args.num_training_steps, warmup)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            beta1=0.9,
            beta2=0.999,
            epsilon=self.args.adam_epsilon,
            parameters=self.model.parameters(),
            weight_decay=self.args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm))

    def print_config(self):
        logger.info('{:^40}'.format("Configuration Arguments"))
        logger.info('{:20}:{}'.format("paddle commit id",
                                      paddle.version.commit))
        for arg in vars(self.args):
            logger.info('{:20}:{}'.format(arg, getattr(self.args, arg)))


def clue_trans_fn(examples, tokenizer, args):
    return convert_clue(
        examples,
        tokenizer=tokenizer,
        label_list=args.label_list,
        max_seq_length=args.max_seq_length)


def clue_batchify_fn(tokenizer, args):
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if args.label_list else "float32")  # label
    ): fn(samples)

    return batchify_fn


def convert_clue(example,
                 label_list,
                 tokenizer=None,
                 is_test=False,
                 max_seq_length=512,
                 **kwargs):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example['label'] = np.array(example["label"], dtype="int64")
        label = example['label']
    # Convert raw text to feature
    if 'keyword' in example:  # CSL
        sentence1 = " ".join(example['keyword'])
        example = {
            'sentence1': sentence1,
            'sentence2': example['abst'],
            'label': example['label']
        }
    elif 'target' in example:  # wsc
        text, query, pronoun, query_idx, pronoun_idx = example['text'], example[
            'target']['span1_text'], example['target']['span2_text'], example[
                'target']['span1_index'], example['target']['span2_index']
        text_list = list(text)
        assert text[pronoun_idx:(pronoun_idx + len(pronoun)
                                 )] == pronoun, "pronoun: {}".format(pronoun)
        assert text[query_idx:(query_idx + len(query)
                               )] == query, "query: {}".format(query)
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

    if tokenizer is None:
        return example
    if 'sentence' in example:
        example = tokenizer(example['sentence'], max_seq_len=max_seq_length)
    elif 'sentence1' in example:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length)
    if not is_test:
        return example['input_ids'], example['token_type_ids'], label
    else:
        return example['input_ids'], example['token_type_ids']


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, mode="dev"):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    accu = metric.accumulate()
    logger.info("%s: eval loss: %.5f, accuracy: %.5f" %
                (mode, np.mean(losses), accu))
    metric.reset()
    model.train()
    return accu


def create_dataloader(dataset,
                      mode='train',
                      batch_size=16,
                      batched=False,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn, batched=False)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)


class CLUE_TRAINING(BaseTrainer):
    def __init__(self, train_ds, dev_ds, model, tokenizer, args):
        super().__init__()
        self.rank = paddle.distributed.get_rank()
        self.train_ds = train_ds
        self.dev_ds = dev_ds
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.dataloader_inner()
        self.prepare_train_config()
        self.print_config()

    def dataloader_inner(self):
        trans_fn = partial(
            clue_trans_fn, tokenizer=self.tokenizer, args=self.args)
        batchify_fn = clue_batchify_fn(self.tokenizer, self.args)

        self.train_dl = self.create_dataloader(
            self.train_ds, "train", self.args.batch_size, batchify_fn, trans_fn)
        self.dev_dl = self.create_dataloader(
            self.dev_ds, "dev", self.args.batch_size, batchify_fn, trans_fn)

    def eval(self):
        pass

    def train(self):
        num_classes = 1 if self.train_ds.label_list == None else len(
            self.train_ds.label_list)

        loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if self.train_ds.label_list else paddle.nn.loss.MSELoss()

        metric = Accuracy()

        if self.args.use_amp:
            scaler = paddle.amp.GradScaler(
                init_loss_scaling=self.args.scale_loss)

        best_dev_acc = 0.0
        corr_test_acc = -1.0
        global_step = 0
        tic_train = time.time()

        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(self.train_dl):
                global_step += 1
                input_ids, segment_ids, labels = batch
                with paddle.amp.auto_cast(
                        bool(self.args.use_amp),
                        custom_white_list=["layer_norm", "softmax", "gelu"], ):
                    logits = self.model(input_ids, segment_ids)
                    loss = loss_fct(logits, labels)

                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.minimize(self.optimizer, loss)
                else:
                    loss.backward()
                    self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.clear_grad()

                if global_step % self.args.logging_steps == 0:
                    logger.info(
                        "global step %d/%d, epoch: %d, batch: %d, acc: %.5f, loss: %f, lr: %.10f, speed: %.4f step/s"
                        % (global_step, self.args.num_training_steps, epoch,
                           step, metric.accumulate(), loss,
                           self.optimizer.get_lr(),
                           self.args.logging_steps / (time.time() - tic_train)))
                    metric.reset()
                    tic_train = time.time()
                if global_step % self.args.valid_steps == 0 or global_step == self.args.num_training_steps:
                    tic_eval = time.time()
                    metric.reset()
                    if self.dev_dl is not None:
                        dev_acc = evaluate(self.model, loss_fct, metric,
                                           self.dev_dl, "dev")
                    else:
                        dev_acc = -1.0
                    metric.reset()
                    test_acc = -1
                    metric.reset()

                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        corr_test_acc = test_acc

                if global_step >= self.args.num_training_steps:
                    logger.info("best_dev_acc: {:.6f}".format(best_dev_acc))
                    logger.info("corr_test_acc: {:.6f}".format(corr_test_acc))
                    return

        logger.info("best_dev_acc: {:.6f}".format(best_dev_acc))
        logger.info("corr_test_acc: {:.6f}".format(corr_test_acc))
