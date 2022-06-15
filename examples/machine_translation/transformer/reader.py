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


def create_data_loader(args, places=None):
    if args.train_file is not None and args.dev_file is not None:
        datasets = load_dataset('wmt14ende',
                                data_files=[args.train_file, args.dev_file],
                                splits=('train', 'dev'))
    elif args.train_file is None and args.dev_file is None:
        datasets = load_dataset('wmt14ende', splits=('train', 'dev'))
    else:
        raise ValueError(
            "--train_file and --dev_file must be both or neither set. ")

    if args.vocab_file is not None:
        src_vocab = Vocab.load_vocabulary(filepath=args.vocab_file,
                                          unk_token=args.unk_token,
                                          bos_token=args.bos_token,
                                          eos_token=args.eos_token)
    elif not args.benchmark:
        src_vocab = Vocab.load_vocabulary(**datasets[0].vocab_info["bpe"])
    else:
        src_vocab = Vocab.load_vocabulary(**datasets[0].vocab_info["benchmark"])
    trg_vocab = src_vocab

    padding_vocab = (
        lambda x:
        (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor)
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))

    def convert_samples(sample):
        source = sample[args.src_lang].split()
        target = sample[args.trg_lang].split()

        source = src_vocab.to_indices(source)
        target = trg_vocab.to_indices(target)

        return source, target

    data_loaders = [(None)] * 2
    for i, dataset in enumerate(datasets):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(min_max_filer, max_len=args.max_length))
        batch_sampler = TransformerBatchSampler(
            dataset=dataset,
            batch_size=args.batch_size,
            pool_size=args.pool_size,
            sort_type=args.sort_type,
            shuffle=args.shuffle,
            shuffle_batch=args.shuffle_batch,
            use_token_batch=True,
            max_length=args.max_length,
            distribute_mode=True if i == 0 else False,
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            pad_seq=args.pad_seq,
            bsz_multi=args.bsz_multi)

        data_loader = DataLoader(dataset=dataset,
                                 places=places,
                                 batch_sampler=batch_sampler,
                                 collate_fn=partial(prepare_train_input,
                                                    bos_idx=args.bos_idx,
                                                    eos_idx=args.eos_idx,
                                                    pad_idx=args.bos_idx,
                                                    pad_seq=args.pad_seq,
                                                    dtype=args.input_dtype),
                                 num_workers=args.num_workers)
        data_loaders[i] = (data_loader)
    return data_loaders


def create_infer_loader(args):
    if args.test_file is not None:
        dataset = load_dataset('wmt14ende',
                               data_files=[args.test_file],
                               splits=['test'])
    else:
        dataset = load_dataset('wmt14ende', splits=('test'))

    if args.vocab_file is not None:
        src_vocab = Vocab.load_vocabulary(filepath=args.vocab_file,
                                          unk_token=args.unk_token,
                                          bos_token=args.bos_token,
                                          eos_token=args.eos_token)
    elif not args.benchmark:
        src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["bpe"])
    else:
        src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["benchmark"])
    trg_vocab = src_vocab

    padding_vocab = (
        lambda x:
        (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor)
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

    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=partial(prepare_infer_input,
                                                bos_idx=args.bos_idx,
                                                eos_idx=args.eos_idx,
                                                pad_idx=args.bos_idx,
                                                pad_seq=args.pad_seq,
                                                dtype=args.input_dtype),
                             num_workers=args.num_workers,
                             return_list=True)
    return data_loader, trg_vocab.to_tokens


def adapt_vocab_size(args):
    if args.vocab_file is not None:
        src_vocab = Vocab.load_vocabulary(filepath=args.vocab_file,
                                          unk_token=args.unk_token,
                                          bos_token=args.bos_token,
                                          eos_token=args.eos_token)
    else:
        dataset = load_dataset('wmt14ende', splits=('test'))
        if not args.benchmark:
            src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["bpe"])
        else:
            src_vocab = Vocab.load_vocabulary(**dataset.vocab_info["benchmark"])
    trg_vocab = src_vocab

    padding_vocab = (
        lambda x:
        (x + args.pad_factor - 1) // args.pad_factor * args.pad_factor)
    args.src_vocab_size = padding_vocab(len(src_vocab))
    args.trg_vocab_size = padding_vocab(len(trg_vocab))


def prepare_train_input(insts,
                        bos_idx,
                        eos_idx,
                        pad_idx,
                        pad_seq=1,
                        dtype="int64"):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx, dtype=dtype)
    src_max_len = (max([len(inst[0])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    trg_max_len = (max([len(inst[1])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])
    trg_word = word_pad([[bos_idx] + inst[1] + [pad_idx] *
                         (trg_max_len - 1 - len(inst[1])) for inst in insts])
    lbl_word = np.expand_dims(word_pad([
        inst[1] + [eos_idx] + [pad_idx] * (trg_max_len - 1 - len(inst[1]))
        for inst in insts
    ]),
                              axis=2)

    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


def prepare_infer_input(insts,
                        bos_idx,
                        eos_idx,
                        pad_idx,
                        pad_seq=1,
                        dtype="int64"):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx, dtype=dtype)
    src_max_len = (max([len(inst[0])
                        for inst in insts]) + pad_seq) // pad_seq * pad_seq
    src_word = word_pad([
        inst[0] + [eos_idx] + [pad_idx] * (src_max_len - 1 - len(inst[0]))
        for inst in insts
    ])

    return [
        src_word,
    ]


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class SentenceBatchCreator(object):

    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator(object):

    def __init__(self, batch_size, bsz_multi=1):
        self._batch = []
        self.max_len = -1
        self._batch_size = batch_size
        self._bsz_multi = bsz_multi

    def append(self, info):
        cur_len = info.max_len
        max_len = max(self.max_len, cur_len)
        if max_len * (len(self._batch) + 1) > self._batch_size:
            # Make sure the batch size won't be empty.
            mode_len = max(
                len(self._batch) // self._bsz_multi * self._bsz_multi,
                len(self._batch) % self._bsz_multi)
            result = self._batch[:mode_len]
            self._batch = self._batch[mode_len:]
            self._batch.append(info)
            self.max_len = max([b.max_len for b in self._batch])
            return result
        else:
            self.max_len = max_len
            self._batch.append(info)

    @property
    def batch(self):
        return self._batch


class SampleInfo(object):

    def __init__(self, i, lens, pad_seq=1):
        self.i = i
        # Take bos and eos into account
        self.min_len = min(lens[0], lens[1]) + 1
        self.max_len = (max(lens[0], lens[1]) + pad_seq) // pad_seq * pad_seq
        self.seq_max_len = max(lens[0], lens[1]) + 1
        self.src_len = lens[0] + 1
        self.trg_len = lens[1] + 1


class TransformerBatchSampler(BatchSampler):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_size=10000,
                 sort_type=SortType.NONE,
                 min_length=0,
                 max_length=100,
                 shuffle=False,
                 shuffle_batch=False,
                 use_token_batch=False,
                 clip_last_batch=False,
                 distribute_mode=True,
                 seed=0,
                 world_size=1,
                 rank=0,
                 pad_seq=1,
                 bsz_multi=8):
        for arg, value in locals().items():
            if arg != "self":
                setattr(self, "_" + arg, value)
        self._random = np.random
        self._random.seed(seed)
        # for multi-devices
        self._distribute_mode = distribute_mode
        self._nranks = world_size
        self._local_rank = rank
        self._sample_infos = []
        for i, data in enumerate(self._dataset):
            lens = [len(data[0]), len(data[1])]
            self._sample_infos.append(SampleInfo(i, lens, self._pad_seq))

    def __iter__(self):
        # global sort or global shuffle
        if self._sort_type == SortType.GLOBAL:
            infos = sorted(self._sample_infos, key=lambda x: x.trg_len)
            infos = sorted(infos, key=lambda x: x.src_len)
        else:
            if self._shuffle:
                infos = self._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._sample_infos

            if self._sort_type == SortType.POOL:
                reverse = True
                for i in range(0, len(infos), self._pool_size):
                    # To avoid placing short next to long sentences
                    reverse = not reverse
                    infos[i:i + self._pool_size] = sorted(
                        infos[i:i + self._pool_size],
                        key=lambda x: x.seq_max_len,
                        reverse=reverse)

        batches = []
        batch_creator = TokenBatchCreator(
            self._batch_size,
            self._bsz_multi) if self._use_token_batch else SentenceBatchCreator(
                self._batch_size * self._nranks)

        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        if self._shuffle_batch:
            self._random.shuffle(batches)

        if not self._use_token_batch:
            # When producing batches according to sequence number, to confirm
            # neighbor batches which would be feed and run parallel have similar
            # length (thus similar computational cost) after shuffle, we as take
            # them as a whole when shuffling and split here
            batches = [[
                batch[self._batch_size * i:self._batch_size * (i + 1)]
                for i in range(self._nranks)
            ] for batch in batches]
            batches = list(itertools.chain.from_iterable(batches))
        self.batch_number = (len(batches) + self._nranks - 1) // self._nranks

        # for multi-device
        for batch_id, batch in enumerate(batches):
            if not self._distribute_mode or (batch_id % self._nranks
                                             == self._local_rank):
                batch_indices = [info.i for info in batch]
                yield batch_indices
        if self._distribute_mode and len(batches) % self._nranks != 0:
            if self._local_rank >= len(batches) % self._nranks:
                # use previous data to pad
                yield batch_indices

    def __len__(self):
        if hasattr(self, "batch_number"):  #
            return self.batch_number
        if not self._use_token_batch:
            batch_number = (len(self._dataset) + self._batch_size * self._nranks
                            - 1) // (self._batch_size * self._nranks)
        else:
            # For uncertain batch number, the actual value is self.batch_number
            batch_number = sys.maxsize
        return batch_number
