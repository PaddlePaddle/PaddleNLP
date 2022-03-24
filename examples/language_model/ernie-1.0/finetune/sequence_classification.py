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
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.metric import Accuracy
import numpy as np

import paddlenlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger

from paddlenlp.trainer.trainer_base import TrainerBase, Trainer


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    if "text_b" in example.keys():
        text = example["text_a"]
        text_pair = example["text_b"]
    else:
        text = example["text"]
        text_pair = None

    encoded_inputs = tokenizer(
        text=text, text_pair=text_pair, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if is_test:
        return input_ids, token_type_ids
    label = np.array([example["label"]], dtype="int64")
    # return input_ids, token_type_ids, label
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "labels": label
    }


def seq_trans_fn(example, tokenizer, args):
    return convert_example(
        example, tokenizer=tokenizer, max_seq_length=args.max_seq_length)


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


class Dict(object):
    def __init__(self, fn):
        assert isinstance(fn, (dict)), 'Input pattern not understood. The input of Dict must be a dict with key of input column name and value of collate_fn ' \
                                   'Received fn=%s' % (str(fn))

        self._fn = fn

        for col_name, ele_fn in self._fn.items():
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                col_name, str(type(ele_fn)))

    def __call__(self, data):

        ret = {}
        for col_name, ele_fn in self._fn.items():
            result = ele_fn([ele[col_name] for ele in data])
            ret[col_name] = result

        return ret


def clue_batchify_fn_dict(tokenizer, args):
    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        "labels": Stack(dtype="int64" if args.label_list else "float32")  # label
    }): fn(samples)

    return batchify_fn


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


class ClueTrainer(TrainerBase):
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
        trans_fn = partial(
            clue_trans_fn, tokenizer=self.tokenizer, args=self.args)
        batchify_fn = clue_batchify_fn(self.tokenizer, self.args)

        self.train_dl = self.create_dataloader(
            self.train_ds, "train", self.args.batch_size, batchify_fn, trans_fn)
        self.dev_dl = self.create_dataloader(
            self.dev_ds, "dev", self.args.batch_size, batchify_fn, trans_fn)

        self.test_dl = None

    def eval(self):
        pass

    def train(self):
        num_classes = 1 if self.train_ds.label_list == None else len(
            self.train_ds.label_list)

        loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if self.train_ds.label_list else paddle.nn.loss.MSELoss()

        metric = Accuracy()

        if self.args.fp16:
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
                        bool(self.args.fp16),
                        custom_white_list=["layer_norm", "softmax", "gelu"], ):
                    logits = self.model(input_ids, segment_ids)
                    loss = loss_fct(logits, labels)

                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                if self.args.fp16:
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
                if global_step % self.args.eval_steps == 0 or global_step == self.args.num_training_steps:
                    tic_eval = time.time()
                    metric.reset()
                    if self.dev_dl is not None:
                        dev_acc = evaluate(self.model, loss_fct, metric,
                                           self.dev_dl, "dev")
                    else:
                        dev_acc = -1.0
                    metric.reset()

                    if self.test_dl is not None:
                        test_acc = evaluate(self.model, loss_fct, metric,
                                            self.test_dl, "test")
                    else:
                        test_acc = -1.0
                    metric.reset()

                    logger.info("eval done total : %s s" %
                                (time.time() - tic_eval))
                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        corr_test_acc = test_acc

                    logger.warning(
                        "best_dev_acc: {:.6f}, corr_test_acc: {:.6f}".format(
                            best_dev_acc, corr_test_acc))

                if global_step >= self.args.num_training_steps:
                    return

        logger.warning("best_dev_acc: {:.6f}, corr_test_acc: {:.6f}".format(
            best_dev_acc, corr_test_acc))


class SeqTrainer2(ClueTrainer):
    def dataloader_inner(self):
        trans_fn = partial(
            seq_trans_fn, tokenizer=self.tokenizer, args=self.args)
        batchify_fn = clue_batchify_fn(self.tokenizer, self.args)

        self.train_dl = self.create_dataloader(
            self.train_ds, "train", self.args.batch_size, batchify_fn, trans_fn)
        self.dev_dl = self.create_dataloader(
            self.dev_ds, "dev", self.args.batch_size, batchify_fn, trans_fn)
        self.test_dl = self.create_dataloader(
            self.test_ds, "dev", self.args.batch_size, batchify_fn, trans_fn)


class SeqTrainer(Trainer):
    def __init__(self, train_ds, dev_ds, model, tokenizer, data_args,
                 training_args, *arg, **kwargs):

        trans_fn = partial(seq_trans_fn, tokenizer=tokenizer, args=data_args)
        batchify_fn = clue_batchify_fn_dict(tokenizer, data_args)

        train_ds = train_ds.map(trans_fn)
        dev_ds = dev_ds.map(trans_fn)

        loss_fct = paddle.nn.loss.CrossEntropyLoss(
        ) if train_ds.label_list else paddle.nn.loss.MSELoss()

        def compute_metrics(p):
            preds = p.predictions[0] if isinstance(p.predictions,
                                                   tuple) else p.predictions

            preds = paddle.to_tensor(preds)
            label = paddle.to_tensor(p.label_ids)

            probs = F.softmax(preds, axis=1)
            metric = Accuracy()
            metric.reset()
            result = metric.compute(preds, label)
            metric.update(result)
            accu = metric.accumulate()
            metric.reset()
            return {"accuracy": accu}

        super().__init__(
            model,
            loss_fct,
            training_args,
            batchify_fn,
            train_ds,
            dev_ds,
            tokenizer,
            compute_metrics=compute_metrics)
