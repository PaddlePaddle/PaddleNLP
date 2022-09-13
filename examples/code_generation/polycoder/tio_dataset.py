# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import os.path
import time
import numpy as np
import paddle
from paddle.io import DataLoader
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.utils.log import logger
from paddlenlp.utils.batch_sampler import DistributedBatchSampler


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


class TioDataset(paddle.io.Dataset):

    def __init__(self,
                 ids,
                 idx,
                 los,
                 eos,
                 shf,
                 seq_len,
                 base,
                 bound,
                 cnt,
                 name: str = 'tio'):
        super().__init__()
        self.name, self.len, self.siz = name, cnt + 1, bound - base
        self.ids, self.idx, self.los, self.shf = ids, idx, los, shf
        self.seq_len, self.base, self.bound = seq_len + 1, base, bound
        self.eos = eos
        self.pos_ids = np.arange(0, seq_len, dtype="int64")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ix = self.shf[index % self.siz + self.base]

        tokens = self.ids[self.idx[ix]:self.idx[ix + 1]]
        loss_msk = np.concatenate((np.zeros(self.los[ix] - 1, dtype='float32'),
                                   np.ones(len(tokens) - self.los[ix],
                                           dtype='float32')))
        attention_mask = np.ones(len(tokens), dtype='float32')
        if len(tokens) < self.seq_len:
            loss_msk = np.concatenate(
                (loss_msk, np.zeros(self.seq_len - len(tokens),
                                    dtype='float32')))
            attention_mask = np.concatenate(
                (attention_mask,
                 np.zeros(self.seq_len - len(tokens), dtype='float32')))
            tokens = tokens.tolist() + [self.eos] * (self.seq_len - len(tokens))
        labels, tokens = tokens[1:], tokens[:-1]
        attention_mask = attention_mask[1:]
        labels = np.array(labels, dtype="int64")
        return [tokens, loss_msk, attention_mask, self.pos_ids, labels]


class TioDatasetLoader(object):

    def __init__(self,
                 path,
                 split,
                 seed,
                 seq_len,
                 eos,
                 train_cnt,
                 valid_cnt,
                 test_cnt,
                 build_shf: bool = False):
        for sfx in ['_ids_tio.npy', '_idx_tio.npz']:
            if not os.path.isfile(path + sfx):
                raise ValueError("File Not found, %s" % (path + sfx))

        self.ids = np.load(path + '_ids_tio.npy')
        npz = np.load(path + '_idx_tio.npz')
        self.idx, self.los = npz['idx'], npz['los']

        self.shf = self.shuffle(path, build_shf, seed)
        siz = len(self.los)
        seg = get_train_valid_test_split_(split, siz)

        self.train = TioDataset(self.ids, self.idx, self.los, eos, self.shf,
                                seq_len, seg[0], seg[1], train_cnt, 'train')
        self.valid = TioDataset(self.ids, self.idx, self.los, eos, self.shf,
                                seq_len, seg[1], seg[2], valid_cnt, 'valid')
        self.test = TioDataset(self.ids, self.idx, self.los, eos, self.shf,
                               seq_len, seg[2], seg[3], test_cnt, 'test')

    def shuffle(self,
                path,
                on_build: bool = False,
                seed: int = 19260817) -> object:
        fl = path + '_shf_tio.npy'
        if on_build:
            siz = len(self.los)
            dtyp = np.uint32
            if siz >= (np.iinfo(np.uint32).max - 1):
                dtyp = np.int64
            shf = np.arange(len(self.los), dtype=dtyp)
            np.random.RandomState(seed=seed).shuffle(shf)
            np.save(fl, shf)
            return shf
        while True:
            if not os.path.isfile(fl):
                time.sleep(3)
            else:
                try:
                    np.load(fl, allow_pickle=True, mmap_mode='r')
                    break
                except Exception as e:
                    print(
                        "%s file is still being written or damaged, please wait a moment."
                        % fl)
                    time.sleep(3)
        return np.load(fl)


def create_pretrained_dataset(
    args,
    input_path,
    local_rank,
    data_world_rank,
    data_world_size,
    eos_id,
    worker_init=None,
    max_seq_len=1024,
    places=None,
    data_holders=None,
    pipeline_mode=False,
):
    device_world_size = paddle.distributed.get_world_size()
    device_world_rank = paddle.distributed.get_rank()

    logger.info(
        "The distributed run, total device num:{}, distinct dataflow num:{}.".
        format(device_world_size, data_world_size))

    assert len(input_path) == 1, "GPT only support one dataset for now."
    input_prefix = input_path[0]

    def build(dataset):
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=args.micro_batch_size,
            num_replicas=data_world_size,
            rank=data_world_rank,
            shuffle=False,
            drop_last=True)

        if pipeline_mode:

            def data_gen():
                for data in dataset:
                    yield tuple(
                        [np.expand_dims(np.array(x), axis=0) for x in data])

            data_loader = paddle.fluid.io.DataLoader.from_generator(
                feed_list=data_holders, capacity=70, iterable=False)
            data_loader.set_batch_generator(data_gen, places)
        else:
            data_loader = DataLoader(dataset=dataset,
                                     places=places,
                                     feed_list=data_holders,
                                     batch_sampler=batch_sampler,
                                     num_workers=0,
                                     worker_init_fn=worker_init,
                                     collate_fn=Tuple(Stack(), Stack(), Stack(),
                                                      Stack(), Stack()),
                                     return_list=False)
        return data_loader

    loaders = TioDatasetLoader(
        input_prefix,
        args.split,
        args.seed,
        max_seq_len,
        eos_id,
        args.micro_batch_size * args.max_steps * data_world_size,
        args.micro_batch_size * (args.max_steps // args.eval_freq + 1) *
        args.eval_iters * data_world_size,
        args.micro_batch_size * args.test_iters * data_world_size,
        build_shf=local_rank == 0)

    if pipeline_mode:
        return build(loaders.train), None, None
    return build(loaders.train), build(loaders.valid), build(loaders.test)
