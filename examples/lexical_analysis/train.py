#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import math
import argparse
from functools import partial

import numpy as np
import paddle
from paddlenlp.data import Pad, Tuple, Stack
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.utils.log import logger
import distutils.util

from data import load_dataset, load_vocab, convert_example
from model import BiGruCrf

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--model_save_dir", type=str, default=None, help="The model will be saved in this path.")
parser.add_argument("--epochs", type=int, default=10, help="Corpus iteration num.")
parser.add_argument("--batch_size", type=int, default=300, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu"] ,help="The device to select to train the model, is must be cpu/gpu.")
parser.add_argument("--base_lr", type=float, default=0.001, help="The basic learning rate that affects the entire network.")
parser.add_argument("--crf_lr", type=float, default=0.2, help="The learning rate ratio that affects CRF layers.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension in which a word is embedded.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
parser.add_argument("--logging_steps", type=int, default=10, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
parser.add_argument("--do_eval", type=distutils.util.strtobool, default=True, help="To evaluate the model if True.")
# yapf: enable


@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        token_ids, length, labels = batch
        preds = model(token_ids, length)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(),
                      num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    logger.info("eval precision: %f, recall: %f, f1: %f" %
                (precision, recall, f1_score))
    model.train()
    return precision, recall, f1_score


def train(args):
    paddle.set_device(args.device)

    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    # Create dataset.
    train_ds, test_ds = load_dataset(
        datafiles=(os.path.join(args.data_dir, 'train.tsv'),
                   os.path.join(args.data_dir, 'test.tsv')))

    word_vocab = load_vocab(os.path.join(args.data_dir, 'word.dic'))
    label_vocab = load_vocab(os.path.join(args.data_dir, 'tag.dic'))
    # q2b.dic is used to replace DBC case to SBC case
    normlize_vocab = load_vocab(os.path.join(args.data_dir, 'q2b.dic'))

    trans_func = partial(convert_example,
                         max_seq_len=args.max_seq_len,
                         word_vocab=word_vocab,
                         label_vocab=label_vocab,
                         normlize_vocab=normlize_vocab)
    train_ds.map(trans_func)
    test_ds.map(trans_func)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get("[PAD]", 0), dtype='int64'
            ),  # word_ids
        Stack(dtype='int64'),  # length
        Pad(axis=0, pad_val=label_vocab.get("O", 0), dtype='int64'
            ),  # label_ids
    ): fn(samples)

    # Create sampler for dataloader
    train_sampler = paddle.io.DistributedBatchSampler(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    train_loader = paddle.io.DataLoader(dataset=train_ds,
                                        batch_sampler=train_sampler,
                                        return_list=True,
                                        collate_fn=batchify_fn)

    test_sampler = paddle.io.BatchSampler(dataset=test_ds,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          drop_last=False)
    test_loader = paddle.io.DataLoader(dataset=test_ds,
                                       batch_sampler=test_sampler,
                                       return_list=True,
                                       collate_fn=batchify_fn)

    # Define the model netword and its loss
    model = BiGruCrf(args.emb_dim,
                     args.hidden_size,
                     len(word_vocab),
                     len(label_vocab),
                     crf_lr=args.crf_lr)
    # Prepare optimizer, loss and metric evaluator
    optimizer = paddle.optimizer.Adam(learning_rate=args.base_lr,
                                      parameters=model.parameters())
    chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)

    if args.init_checkpoint:
        if os.path.exists(args.init_checkpoint):
            logger.info("Init checkpoint from %s" % args.init_checkpoint)
            model_dict = paddle.load(args.init_checkpoint)
            model.load_dict(model_dict)
        else:
            logger.info("Cannot init checkpoint from %s which doesn't exist" %
                        args.init_checkpoint)
    logger.info("Start training")
    # Start training
    global_step = 0
    last_step = args.epochs * len(train_loader)
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    max_f1_score = -1
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            train_reader_cost += time.time() - reader_start
            global_step += 1
            token_ids, length, label_ids = batch
            train_start = time.time()
            loss = model(token_ids, length, label_ids)
            avg_loss = paddle.mean(loss)
            train_run_cost += time.time() - train_start
            total_samples += args.batch_size
            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
                    %
                    (global_step, last_step, avg_loss,
                     train_reader_cost / args.logging_steps,
                     (train_reader_cost + train_run_cost) / args.logging_steps,
                     total_samples / args.logging_steps, total_samples /
                     (train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if global_step % args.save_steps == 0 or global_step == last_step:
                if rank == 0:
                    paddle.save(
                        model.state_dict(),
                        os.path.join(args.model_save_dir,
                                     "model_%d.pdparams" % global_step))
                    logger.info("Save %d steps model." % (global_step))
                    if args.do_eval:
                        precision, recall, f1_score = evaluate(
                            model, chunk_evaluator, test_loader)
                        if f1_score > max_f1_score:
                            max_f1_score = f1_score
                            paddle.save(
                                model.state_dict(),
                                os.path.join(args.model_save_dir,
                                             "best_model.pdparams"))
                            logger.info("Save best model.")

            reader_start = time.time()


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
