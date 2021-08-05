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

import logging
import argparse
import os
import time

import numpy as np
import paddle
from paddlenlp.transformers.optimization import LinearDecayWithWarmup

from env import Environment
from data import batchify, TextDataset, Corpus
from model.model import DDParserModel
from model.metric import ParserEvaluator
from model.model_utils import ParserCriterion, decode

# yapf: disable
parser = argparse.ArgumentParser()
# Train
parser.add_argument("--epochs", type=int, default=100, help="Number of epoches for training.")
parser.add_argument("--device", choices=["cpu", "gpu", "xpu"], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint.")
parser.add_argument("--data_path", type=str, default='./data', help="The path of datasets to be loaded.")
parser.add_argument("--max_batch_size", type=int, default=5000, help="Maximum examples of a batch for training.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--encoding_model", choices=["lstm", "lstm-pe", "ernie-1.0", "ernie-tiny", "ernie-gram-zh"], type=str, default="ernie-1.0", help="Select the encoding model.")
parser.add_argument("--clip", type=float, default=1.0, help="The threshold of gradient clip.")
parser.add_argument("--lstm_lr", type=float, default=0.002, help="The Learning rate of lstm encoding model.")
parser.add_argument("--ernie_lr", type=float, default=5e-05, help="The Learning rate of ernie encoding model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
# Preprocess
parser.add_argument("--min_freq", type=int, default=2, help="The minimum frequency of word when construct the vocabulary.")
parser.add_argument("--preprocess", type=bool, default=True, help="Whether to preprocess the dataset.")
parser.add_argument("--fix_len", type=int, default=20)
parser.add_argument("--n_buckets", type=int, default=15, help="Number of buckets to devide the dataset.")
# Decode
parser.add_argument("--tree", type=bool, default=True, help="Ensure the output conforms to the tree structure.")
# Lstm
parser.add_argument("--beta1", type=float, default=0.9, help="The coefficient of adam optimizer for computing running average of gradient.")
parser.add_argument("--beta2", type=float, default=0.9, help="The coefficient of adam optimizer for computing running average of squared-gradient.")
parser.add_argument("--epsilon", type=float, default=1e-12, help="A small float value for numerical stability.")
parser.add_argument("--decay_rate", type=float, default=0.75, help="The decay rate of learning rate.")
parser.add_argument("--feat", choices=["char", "pos"], type=str, default=None, help="The feature representation to use.")
# Ernie
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="Linear warmup proportion over total steps.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
parser.add_argument("--mode", type=str, default="train")
args = parser.parse_args()
# yapf: enable

@paddle.no_grad()
def evaluate(args, model, metric, criterion, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader():
        if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
            words, arcs, rels = batch
            s_arc, s_rel, words = model(words)
        else:
            words, feats, arcs, rels = batch
            s_arc, s_rel, words = model(words, feats)   

        mask = paddle.logical_and(
                paddle.logical_and(words != args.pad_index, words != args.bos_index),
                words != args.eos_index,
        )

        loss = criterion(s_arc, s_rel, arcs, rels, mask)

        losses.append(loss.numpy().item())

        arc_preds, rel_preds = decode(args, s_arc, s_rel, mask)
        metric.update(arc_preds, rel_preds, arcs, rels, mask)
        uas, las = metric.accumulate()

    print("eval loss: %.5f, UAS: %.5f, LAS: %.5f" % (np.mean(losses), uas, las))
    logging.info("eval loss: %.5f, UAS: %.5f, LAS: %.5f" % (np.mean(losses), uas, las))
    model.train()
    metric.reset()

def do_train():
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    env = Environment(args)

    train = Corpus.load("./data/train.txt", env.fields)
    dev = Corpus.load("./data/test.txt", env.fields)

    train_ds = TextDataset(train, env.fields, args.n_buckets)
    dev_ds = TextDataset(dev, env.fields, args.n_buckets)

    train_data_loader = batchify(train_ds, args.max_batch_size, shuffle=True, use_multiprocess=True)
    dev_data_loader = batchify(dev_ds, args.max_batch_size)
    
    if args.encoding_model.startswith("ernie"):
        lr = args.ernie_lr
        model = DDParserModel(args=args, pretrained_model=env.pretrained_model)
    else:
        lr = args.lstm_lr
        model = DDParserModel(args=args)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)
    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(lr, num_training_steps, args.warmup_proportion)
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=args.clip)

    if args.encoding_model.startswith("ernie"):
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=grad_clip,
        )
    else:
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr_scheduler,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            parameters=model.parameters(),
            grad_clip=grad_clip,
        )

    # train
    metric = ParserEvaluator()
    criterion = ParserCriterion()

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for step, inputs in enumerate(train_data_loader(), start=1):
            if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
                words, arcs, rels = inputs
                s_arc, s_rel, words = model(words)
            else:
                words, feats, arcs, rels = inputs
                s_arc, s_rel, words = model(words, feats)
        
            mask = paddle.logical_and(
                paddle.logical_and(words != args.pad_index, words != args.bos_index),
                words != args.eos_index,
            )

            loss = criterion(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            global_step += 1
            if global_step % 100 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss.numpy().item(), 10 / (time.time() - tic_train)))
                tic_train = time.time()

            total_loss += loss.numpy().item()
        if rank == 0:
            save_dir = os.path.join(args.save_dir, "model_epoch_%d" % epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            evaluate(args, model, metric, criterion, dev_data_loader)
            save_param_path = os.path.join(save_dir, "model_state.pdparams")
            paddle.save(model.state_dict(), save_param_path)
            if args.encoding_model.startswith("ernie") or args.encoding_model == "lstm-pe":
                env.tokenizer.save_pretrained(save_dir)                           
        total_loss /= len(train_data_loader)
    
if __name__ == "__main__":
    do_train()