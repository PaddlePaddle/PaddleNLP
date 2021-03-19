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

import sys
import os
import io
import itertools
from functools import partial

import numpy as np
from paddle.io import BatchSampler, DataLoader, Dataset
import paddle.distributed as dist
from paddlenlp.data import Pad, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.data.sampler import SamplerHelper


def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def create_data_loader(args, places=None, use_all_vocab=False):
    data_files = None
    if args.root != "None" and os.path.exists(args.root):
        data_files = {
            'train': (os.path.join(args.root, "train.tok.clean.bpe.33708.en"),
                      os.path.join(args.root, "train.tok.clean.bpe.33708.de")),
            'dev': (os.path.join(args.root, "newstest2013.tok.bpe.33708.en"),
                    os.path.join(args.root, "newstest2013.tok.bpe.33708.de"))
        }

    datasets = load_dataset(
        'wmt14ende', data_files=data_files, splits=('train', 'dev'))
    if use_all_vocab:
        src_vocab = Vocab.load_vocabulary(**datasets[0].vocab_info["bpe"])
    else:
        src_vocab = Vocab.load_vocabulary(**datasets[0].vocab_info["benchmark"])
    trg_vocab = src_vocab

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample[args.src_lang].split()
        target = sample[args.trg_lang].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    def _max_token_fn(current_idx, current_batch_size, tokens_sofar,
                      data_source):
        return max(tokens_sofar,
                   len(data_source[current_idx][0]) + 1,
                   len(data_source[current_idx][1]) + 1)

    def _key(size_so_far, minibatch_len):
        return size_so_far * minibatch_len

    data_loaders = [(None)] * 2
    for i, dataset in enumerate(datasets):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(
                min_max_filer, max_len=args.max_length))

        sampler = SamplerHelper(dataset)

        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda x, data_source: len(data_source[x][0]) + 1)
            trg_key = (lambda x, data_source: len(data_source[x][1]) + 1)
            # Sort twice
            sampler = sampler.sort(key=trg_key).sort(key=src_key)
        else:
            if args.shuffle:
                sampler = sampler.shuffle(seed=args.shuffle_seed)
            max_key = (lambda x, data_source: max(len(data_source[x][0]), len(data_source[x][1])) + 1)
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)

        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=_max_token_fn,
            key=_key)

        if args.shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=args.shuffle_seed)

        if i == 0:
            batch_sampler = batch_sampler.shard()

        data_loader = DataLoader(
            dataset=dataset,
            places=places,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                pad_idx=args.bos_idx,
                pad_seq=args.pad_seq),
            num_workers=0)
        data_loaders[i] = (data_loader)
    return data_loaders


def create_infer_loader(args, use_all_vocab=False):
    data_files = None
    if args.root != "None" and os.path.exists(args.root):
        data_files = {
            'test': (os.path.join(args.root, "newstest2014.tok.bpe.33708.en"),
                     os.path.join(args.root, "newstest2014.tok.bpe.33708.de"))
        }

    dataset = load_dataset('wmt14ende', data_files=data_files, splits=('test'))
    if use_all_vocab:
        src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["bpe"])
    else:
        src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["benchmark"])
    trg_vocab = src_vocab

    padding_vocab = (
        lambda x: (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor
    )
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample[args.src_lang].split()
        target = sample[args.trg_lang].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    batch_sampler = SamplerHelper(dataset).batch(
        batch_size=args.infer_batch_size, drop_last=False)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(
            prepare_infer_input,
            bos_idx=args.bos_idx,
            eos_idx=args.eos_idx,
            pad_idx=args.bos_idx,
            pad_seq=args.pad_seq),
        num_workers=0,
        return_list=True)
    return data_loader, trg_vocab.to_tokens


def prepare_train_input(insts, bos_idx, eos_idx, pad_idx, pad_seq=1):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_max_len = (
        max([len(inst[0]) for inst in insts]) + pad_seq) // pad_seq * pad_seq
    trg_max_len = (
        max([len(inst[1]) for inst in insts]) + pad_seq) // pad_seq * pad_seq
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])
    trg_word = word_pad([[bos_idx] + inst[1] + [pad_idx] *
                         (trg_max_len - 1 - len(inst[1])) for inst in insts])
    lbl_word = np.expand_dims(
        word_pad([
            inst[1] + [eos_idx] + [pad_idx] * (trg_max_len - 1 - len(inst[1]))
            for inst in insts
        ]),
        axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


def prepare_infer_input(insts, bos_idx, eos_idx, pad_idx, pad_seq=1):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_max_len = (
        max([len(inst[0]) for inst in insts]) + pad_seq) // pad_seq * pad_seq
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])

    return [src_word, ]


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"
