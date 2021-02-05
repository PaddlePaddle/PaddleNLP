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

from paddlenlp.datasets import IWSLT15

trans_func_tuple = IWSLT15.get_default_transform_func()


def create_train_loader(args):
    batch_size = args.batch_size
    max_len = args.max_len
    src_vocab, tgt_vocab = IWSLT15.get_vocab()
    bos_id = src_vocab[src_vocab.bos_token]
    eos_id = src_vocab[src_vocab.eos_token]
    pad_id = eos_id

    train_ds, dev_ds = IWSLT15.get_datasets(
        mode=["train", "dev"],
        transform_func=[trans_func_tuple, trans_func_tuple])

    key = (lambda x, data_source: len(data_source[x][0]))
    cut_fn = lambda data: (data[0][:max_len], data[1][:max_len])

    train_ds = train_ds.filter(
        lambda data: (len(data[0]) > 0 and len(data[1]) > 0)).apply(cut_fn)
    dev_ds = dev_ds.filter(
        lambda data: (len(data[0]) > 0 and len(data[1]) > 0)).apply(cut_fn)
    train_batch_sampler = SamplerHelper(train_ds).shuffle().sort(
        key=key, buffer_size=batch_size * 20).batch(batch_size=batch_size)

    dev_batch_sampler = SamplerHelper(dev_ds).sort(
        key=key, buffer_size=batch_size * 20).batch(batch_size=batch_size)

    train_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=partial(
            prepare_train_input, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id))

    dev_loader = paddle.io.DataLoader(
        dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=partial(
            prepare_train_input, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id))

    return train_loader, dev_loader, len(src_vocab), len(tgt_vocab), pad_id


def create_infer_loader(args):
    batch_size = args.batch_size
    max_len = args.max_len
    trans_func_tuple = IWSLT15.get_default_transform_func()
    test_ds = IWSLT15.get_datasets(
        mode=["test"], transform_func=[trans_func_tuple])
    src_vocab, tgt_vocab = IWSLT15.get_vocab()
    bos_id = src_vocab[src_vocab.bos_token]
    eos_id = src_vocab[src_vocab.eos_token]
    pad_id = eos_id

    test_batch_sampler = SamplerHelper(test_ds).batch(batch_size=batch_size)

    test_loader = paddle.io.DataLoader(
        test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=partial(
            prepare_infer_input, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id))
    return test_loader, len(src_vocab), len(tgt_vocab), bos_id, eos_id


def prepare_infer_input(insts, bos_id, eos_id, pad_id):
    insts = [([bos_id] + inst[0] + [eos_id], [bos_id] + inst[1] + [eos_id])
             for inst in insts]
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    return src, src_length


def prepare_train_input(insts, bos_id, eos_id, pad_id):
    # Add eos token id and bos token id.
    insts = [([bos_id] + inst[0] + [eos_id], [bos_id] + inst[1] + [eos_id])
             for inst in insts]
    # Pad sequence using eos id.
    src, src_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[0] for inst in insts])
    tgt, tgt_length = Pad(pad_val=pad_id, ret_length=True)(
        [inst[1] for inst in insts])
    tgt_mask = (tgt[:, :-1] != pad_id).astype("float32")
    return src, src_length, tgt[:, :-1], tgt[:, 1:, np.newaxis], tgt_mask
