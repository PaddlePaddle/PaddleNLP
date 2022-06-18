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

import io
import os

from functools import partial
import numpy as np

import paddle
from paddlenlp.data import Vocab, Pad
from paddlenlp.data import SamplerHelper
from paddlenlp.datasets import load_dataset


def create_data_loader(args):
    batch_size = args.batch_size
    max_len = args.max_len
    if args.dataset == 'yahoo':
        train_ds, dev_ds, test_ds = load_dataset('yahoo_answer_100k',
                                                 splits=('train', 'valid',
                                                         'test'))
        vocab = Vocab.load_vocabulary(**train_ds.vocab_info)
    else:
        train_ds, dev_ds, test_ds = load_dataset('ptb',
                                                 splits=('train', 'valid',
                                                         'test'))
        examples = [
            train_ds[i]['sentence'].split() for i in range(len(train_ds))
        ]
        vocab = Vocab.build_vocab(examples)

    vocab_size = len(vocab)
    bos_id = vocab_size
    eos_id = vocab_size + 1
    pad_id = vocab_size + 1

    def convert_example(example):
        features = vocab.to_indices(example['sentence'].split()[:max_len])
        return features

    key = (lambda x, data_source: len(data_source[x]))
    # Truncate and convert example to ids
    train_ds = train_ds.map(convert_example, lazy=False)
    dev_ds = dev_ds.map(convert_example, lazy=False)
    test_ds = test_ds.map(convert_example, lazy=False)

    train_batch_sampler = SamplerHelper(train_ds).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(batch_size=batch_size)

    dev_batch_sampler = SamplerHelper(dev_ds).sort(
        key=key, buffer_size=batch_size * 20).batch(batch_size=batch_size)

    test_batch_sampler = SamplerHelper(dev_ds).sort(
        key=key, buffer_size=batch_size * 20).batch(batch_size=batch_size)

    train_loader = paddle.io.DataLoader(train_ds,
                                        batch_sampler=train_batch_sampler,
                                        collate_fn=partial(prepare_train_input,
                                                           bos_id=bos_id,
                                                           eos_id=eos_id,
                                                           pad_id=pad_id))

    dev_loader = paddle.io.DataLoader(dev_ds,
                                      batch_sampler=dev_batch_sampler,
                                      collate_fn=partial(prepare_train_input,
                                                         bos_id=bos_id,
                                                         eos_id=eos_id,
                                                         pad_id=pad_id))

    test_loader = paddle.io.DataLoader(test_ds,
                                       batch_sampler=dev_batch_sampler,
                                       collate_fn=partial(prepare_train_input,
                                                          bos_id=bos_id,
                                                          eos_id=eos_id,
                                                          pad_id=pad_id))

    return train_loader, dev_loader, test_loader, vocab, bos_id, pad_id, len(
        train_ds)


def prepare_train_input(insts, bos_id, eos_id, pad_id):
    # Add eos token id and bos token id.
    src = [[bos_id] + inst + [eos_id] for inst in insts]
    trg = [inst[:-1] for inst in insts]
    label = [inst[1:] for inst in insts]

    # Pad sequence using eos id.
    src, src_length = Pad(pad_val=pad_id, ret_length=True,
                          dtype="int64")([ids for ids in src])
    trg, trg_length = Pad(pad_val=pad_id, ret_length=True,
                          dtype="int64")([ids for ids in trg])
    label, _ = Pad(pad_val=pad_id, ret_length=True,
                   dtype="int64")([ids for ids in label])

    label = np.array(label)
    label = label.reshape((label.shape[0], label.shape[1], 1))
    return src, src_length, trg, trg_length, label
