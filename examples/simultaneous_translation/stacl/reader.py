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

from functools import partial
from paddle.io import DataLoader
from paddlenlp.data import Vocab, Pad
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.datasets import load_dataset


def read(src_tgt_file, only_src=False):
    with open(src_tgt_file, 'r', encoding='utf8') as src_tgt_f:
        for line in src_tgt_f:
            line = line.strip('\n')
            if not line:
                continue
            line_split = line.split('\t')
            if only_src:
                yield {"src": line_split[0]}
            else:
                if len(line_split) != 2:
                    continue
                yield {"src": line_split[0], "trg": line_split[1]}


def min_max_filer(data, max_len, min_len=0):
    # 1 for special tokens.
    data_min_len = min(len(data[0]), len(data[1])) + 1
    data_max_len = max(len(data[0]), len(data[1])) + 1
    return (data_min_len >= min_len) and (data_max_len <= max_len)


def create_data_loader(args, places=None):
    data_files = {'train': args.training_file, 'dev': args.validation_file}

    datasets = [
        load_dataset(read, src_tgt_file=filename, lazy=False)
        for split, filename in data_files.items()
    ]

    src_vocab = Vocab.load_vocabulary(args.src_vocab_fpath,
                                      bos_token=args.special_token[0],
                                      eos_token=args.special_token[1],
                                      unk_token=args.special_token[2])
    trg_vocab = Vocab.load_vocabulary(args.trg_vocab_fpath,
                                      bos_token=args.special_token[0],
                                      eos_token=args.special_token[1],
                                      unk_token=args.special_token[2])

    args.src_vocab_size = len(src_vocab)
    args.trg_vocab_size = len(trg_vocab)

    def convert_samples(sample):
        source = [item.strip() for item in sample['src'].split()]
        target = [item.strip() for item in sample['trg'].split()]

        source = src_vocab.to_indices(source) + [args.eos_idx]
        target = [args.bos_idx] + \
                 trg_vocab.to_indices(target) + [args.eos_idx]

        return source, target

    data_loaders = [(None)] * 2
    for i, dataset in enumerate(datasets):
        dataset = dataset.map(convert_samples, lazy=False).filter(
            partial(min_max_filer, max_len=args.max_length))

        sampler = SamplerHelper(dataset)

        if args.sort_type == SortType.GLOBAL:
            src_key = (lambda x, data_source: len(data_source[x][0]))
            trg_key = (lambda x, data_source: len(data_source[x][1]))
            # Sort twice
            sampler = sampler.sort(key=trg_key).sort(key=src_key)
        else:
            if args.shuffle:
                sampler = sampler.shuffle(seed=args.random_seed)
            max_key = (lambda x, data_source: max(len(data_source[x][0]),
                                                  len(data_source[x][1])))
            if args.sort_type == SortType.POOL:
                sampler = sampler.sort(key=max_key, buffer_size=args.pool_size)

        batch_size_fn = lambda new, count, sofar, data_source: max(
            sofar, len(data_source[new][0]), len(data_source[new][1]))
        batch_sampler = sampler.batch(
            batch_size=args.batch_size,
            drop_last=False,
            batch_size_fn=batch_size_fn,
            key=lambda size_so_far, minibatch_len: size_so_far * minibatch_len)

        if args.shuffle_batch:
            batch_sampler = batch_sampler.shuffle(seed=args.random_seed)

        if i == 0:
            batch_sampler = batch_sampler.shard()

        data_loader = DataLoader(dataset=dataset,
                                 places=places,
                                 batch_sampler=batch_sampler,
                                 collate_fn=partial(prepare_train_input,
                                                    pad_idx=args.bos_idx),
                                 num_workers=0)

        data_loaders[i] = (data_loader)

    return data_loaders


def create_infer_loader(args, places=None):
    data_files = {
        'test': args.predict_file,
    }
    dataset = load_dataset(read,
                           src_tgt_file=data_files['test'],
                           only_src=True,
                           lazy=False)

    src_vocab = Vocab.load_vocabulary(args.src_vocab_fpath,
                                      bos_token=args.special_token[0],
                                      eos_token=args.special_token[1],
                                      unk_token=args.special_token[2])

    trg_vocab = Vocab.load_vocabulary(args.trg_vocab_fpath,
                                      bos_token=args.special_token[0],
                                      eos_token=args.special_token[1],
                                      unk_token=args.special_token[2])

    args.src_vocab_size = len(src_vocab)
    args.trg_vocab_size = len(trg_vocab)

    def convert_samples(sample):
        source = [item.strip() for item in sample['src'].split()]
        source = src_vocab.to_indices(source) + [args.eos_idx]
        target = [args.bos_idx]
        return source, target

    dataset = dataset.map(convert_samples, lazy=False)

    batch_sampler = SamplerHelper(dataset).batch(batch_size=args.batch_size,
                                                 drop_last=False)

    data_loader = DataLoader(dataset=dataset,
                             places=places,
                             batch_sampler=batch_sampler,
                             collate_fn=partial(prepare_infer_input,
                                                pad_idx=args.bos_idx),
                             num_workers=0,
                             return_list=True)

    return data_loader, trg_vocab.to_tokens


def prepare_train_input(insts, pad_idx):
    """
    Put all padded data needed by training into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] for inst in insts])
    trg_word = word_pad([inst[1][:-1] for inst in insts])
    lbl_word = word_pad([inst[1][1:] for inst in insts])
    data_inputs = [src_word, trg_word, lbl_word]

    return data_inputs


def prepare_infer_input(insts, pad_idx):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    word_pad = Pad(pad_idx)
    src_word = word_pad([inst[0] for inst in insts])

    return [
        src_word,
    ]


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"
