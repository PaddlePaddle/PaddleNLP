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
import collections

import os
import random
from functools import partial
import time

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset
from paddlenlp.transformers import ErnieDocModel, ErnieDocForSequenceClassification, BPETokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack

from batching import BatchiFy

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="", help="pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=512, help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=7e-5, help="Learning rate used to train.")
parser.add_argument("--max_steps", default=10000, type=int, help="Max training steps to train.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention ffn model dropout.")
parser.add_argument("--hidden_dropout_prob", type=float, default=0.0, help="The dropout rate for the embedding pooler.")
parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Select cpu, gpu devices to train model.")
parser.add_argument("--seed", type=int, default=8, help="Random seed for initialization.")
parser.add_argument("--memory_length", type=int, default=128, help="Random seed for initialization.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--num_labels", default=2, type=int, help="The number of labels.")

# yapf: enable
args = parser.parse_args()


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def init_memory(batch_size, memory_length, d_model, n_layers):
    return [
        np.zeros(
            [batch_size, memory_length, d_model], dtype="float32")
        for _ in range(n_layers)
    ]


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, memories0):
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
    memories = memories0
    for batch in data_loader:
        # TODO(zhoushunjie): need to modify
        input_ids, position_ids, token_type_ids, attn_mask, labels, gather_idxs, need_cal_loss = batch
        logits, memories = model(input_ids, memories, token_type_ids,
                                 position_ids, attn_mask)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    logger.info("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()


def create_dataloader(dataset, mode='train', batch_size=1, batchify_fn=None):
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
        return_list=True)


def do_train(args):
    paddle.set_device(args.device)
    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    set_seed(args)
    # if rank == 0:
    #     if os.path.exists(args.model_name_or_path):
    #         print("init checkpoint from %s" % args.model_name_or_path)
    # TODO(zhoushunjie):need to use pretraining params.
    # model = ErnieDocForSequenceClassification.from_pretrained(args.model_name_or_path)
    ernie_doc = ErnieDocModel(
        **ErnieDocModel.pretrained_init_configuration["ernie-doc-base-en"])
    model = ErnieDocForSequenceClassification(ernie_doc, args.num_labels)
    model_config = model.ernie_doc.config
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    tokenizer = BPETokenizer("vocab.txt")

    train_ds, test_ds = load_dataset("imdb", splits=["train", "test"])

    batchify_fn = BatchiFy(args.batch_size, tokenizer, trainer_num)
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn)
    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    memories = paddle.to_tensor(
        init_memory(args.batch_size, args.memory_length, model_config[
            "hidden_size"], model_config["num_hidden_layers"]))

    for epoch in range(args.epochs):
        print("epoch: {}, number of train_data_loader: {}".format(
            epoch, len(train_data_loader)))
        for step, batch in enumerate(train_data_loader):
            input_ids, position_ids, token_type_ids, input_mask, labels, gather_idx, need_cal_loss = batch
            print("step {}:".format(step))
            print("input_ids: ", input_ids)
            print("position_ids: ", position_ids)
            print("token_type_ids: ", token_type_ids)
            print("input_mask: ", input_mask)
            print("labels: ", labels)
            print("gather_idx: ", gather_idx)
            print("need_cal_loss:", need_cal_loss)
            # logits, memories = model(input_ids, memories, token_type_ids, position_ids, attn_mask)


if __name__ == "__main__":
    do_train(args)
