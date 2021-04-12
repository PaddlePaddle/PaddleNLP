# -*- coding: UTF-8 -*-
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
import math
import argparse

import numpy as np
import paddle
from paddle.static import InputSpec
from paddlenlp.data import Pad, Tuple, Stack
from paddlenlp.metrics import ChunkEvaluator

from data import LacDataset
from model import BiGruCrf

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--data_dir", type=str, default=None, help="The folder where the dataset is located.")
parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to init model.")
parser.add_argument("--batch_size", type=int, default=300, help="The number of sequences contained in a mini-batch.")
parser.add_argument("--max_seq_len", type=int, default=64, help="Number of words of the longest seqence.")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
parser.add_argument("--emb_dim", type=int, default=128, help="The dimension in which a word is embedded.")
parser.add_argument("--hidden_size", type=int, default=128, help="The number of hidden nodes in the GRU layer.")
args = parser.parse_args()
# yapf: enable


def evaluate(args):
    paddle.set_device(args.device)

    # create dataset.
    test_dataset = LacDataset(args.data_dir, mode='test')
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0, dtype='int64'),  # word_ids
        Stack(dtype='int64'),  # length
        Pad(axis=0, pad_val=0, dtype='int64'),  # label_ids
    ): fn(samples)

    # Create sampler for dataloader
    test_sampler = paddle.io.BatchSampler(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_sampler=test_sampler,
        return_list=True,
        collate_fn=batchify_fn)

    # Define the model network and metric evaluator
    network = BiGruCrf(args.emb_dim, args.hidden_size, test_dataset.vocab_size,
                       test_dataset.num_labels)
    inputs = InputSpec(shape=(-1, ), dtype="int64", name='inputs')
    lengths = InputSpec(shape=(-1, ), dtype="int64", name='lengths')
    model = paddle.Model(network, inputs=[inputs, lengths])
    chunk_evaluator = ChunkEvaluator(
        label_list=test_dataset.label_vocab.keys(), suffix=True)
    model.prepare(None, None, chunk_evaluator)

    # Load the model and start predicting
    model.load(args.init_checkpoint)
    model.evaluate(
        eval_data=test_loader,
        batch_size=args.batch_size,
        log_freq=100,
        verbose=2, )


if __name__ == '__main__':
    args = parser.parse_args()
    evaluate(args)
