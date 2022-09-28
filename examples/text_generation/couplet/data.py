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


def convert_example(example, vocab):
    pad_id = vocab[vocab.eos_token]
    bos_id = vocab[vocab.bos_token]
    eos_id = vocab[vocab.eos_token]

    source = [bos_id] + vocab.to_indices(
        example['first'].split('\x02')) + [eos_id]
    target = [bos_id] + vocab.to_indices(
        example['second'].split('\x02')) + [eos_id]
    return source, target


def create_train_loader(batch_size=128):
    train_ds = load_dataset('couplet', splits='train')
    vocab = Vocab.load_vocabulary(**train_ds.vocab_info)
    pad_id = vocab[vocab.eos_token]
    trans_func = partial(convert_example, vocab=vocab)
    train_ds = train_ds.map(trans_func, lazy=False)
    train_batch_sampler = SamplerHelper(train_ds).shuffle().batch(
        batch_size=batch_size)

    train_loader = paddle.io.DataLoader(train_ds,
                                        batch_sampler=train_batch_sampler,
                                        collate_fn=partial(prepare_input,
                                                           pad_id=pad_id))
    return train_loader, vocab


def create_infer_loader(batch_size=128):
    test_ds = load_dataset('couplet', splits='test')
    vocab = Vocab.load_vocabulary(**test_ds.vocab_info)
    pad_id = vocab[vocab.eos_token]
    trans_func = partial(convert_example, vocab=vocab)
    test_ds = test_ds.map(trans_func, lazy=False)
    test_batch_sampler = SamplerHelper(test_ds).batch(batch_size=batch_size)

    test_loader = paddle.io.DataLoader(test_ds,
                                       batch_sampler=test_batch_sampler,
                                       collate_fn=partial(prepare_input,
                                                          pad_id=pad_id))
    return test_loader, vocab


def prepare_input(insts, pad_id):
    src, src_length = Pad(pad_val=pad_id,
                          ret_length=True)([inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True,
                          dtype="int64")([inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype(paddle.get_default_dtype())
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask
