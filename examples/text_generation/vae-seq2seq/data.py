# -*- coding: utf-8 -*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import io

from collections import Counter
import numpy as np

import paddle


class VAEDataset(paddle.io.Dataset):
    def __init__(self,
                 dataset,
                 batch_size,
                 max_seq_len=128,
                 sort_cache=False,
                 cache_num=20,
                 max_vocab_cnt=-1,
                 PAD_ID=0,
                 BOS_ID=1,
                 EOS_ID=2,
                 UNK_ID=3,
                 mode='train'):
        super(VAEDataset, self).__init__()
        self.PAD_ID = PAD_ID
        self.BOS_ID = BOS_ID
        self.EOS_ID = EOS_ID
        self.UNK_ID = UNK_ID
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.train_data_len = 0
        train_data, valid_data, test_data = self.read_raw_data(dataset,
                                                               max_vocab_cnt)
        if mode == 'train':
            raw_data = train_data
            self.train_data_len = len(raw_data)
        elif mode == 'eval':
            raw_data = valid_data
        elif mode == 'test':
            raw_data = test_data
        else:
            raise Exception("Only train|eval|test is supported")

        if self.max_seq_len <= 2:
            raise Exception("The minimum of max sequence length is 3.")

        self.data = self.generate_dataset(
            raw_data,
            batch_size,
            sort_cache=sort_cache,
            cache_num=cache_num,
            mode=mode)

    def read_all_line(self, filename):
        data = []
        with io.open(filename, "r", encoding='utf-8') as f:
            for line in f.readlines():
                data.append(line.strip())
        return data

    def build_vocab(self, train_file, max_vocab_cnt=-1, vocab_file=None):
        vocab_dict = {
            "<pad>": self.PAD_ID,
            "<bos>": self.BOS_ID,
            "<eos>": self.EOS_ID,
            "<unk>": self.UNK_ID
        }
        ids = 4
        if vocab_file:
            with io.open(vocab_file, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    vocab_dict[line.strip()] = ids
                    ids += 1
        else:
            lines = self.read_all_line(train_file)
            all_words = []
            for line in lines:
                all_words.extend(line.split())
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = min(len(vocab_count), max_vocab_cnt)
            for voc, fre in vocab_count[0:raw_vocab_size]:
                if voc not in vocab_dict:
                    vocab_dict[voc] = ids
                    ids += 1
        self.vocab_size = ids
        return vocab_dict

    def corpus_to_token_ids(self, corpus_path, vocab):
        corpus_ids = []
        with open(corpus_path, "r", encoding="utf-8") as f_corpus:
            for line in f_corpus.readlines():
                tokens = line.strip().split()
                if len(tokens) == 0:
                    continue
                ids = [self.BOS_ID]
                if self.mode == 'train' and len(tokens) + 2 > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len - 2]
                ids.extend(
                    [vocab[w] if w in vocab else self.UNK_ID for w in tokens])
                ids.append(self.EOS_ID)
                corpus_ids.append(ids)

        return corpus_ids

    def read_raw_data(self, dataset, max_vocab_cnt=-1):
        if dataset == 'yahoo':
            dataset_name = 'yahoo-answer-100k'
        else:
            dataset_name = os.path.join('simple-examples', 'data')
        data_path = os.path.join("data", dataset_name)
        train_file = os.path.join(data_path, dataset + ".train.txt")
        valid_file = os.path.join(data_path, dataset + ".valid.txt")
        test_file = os.path.join(data_path, dataset + ".test.txt")
        vocab_file = None
        if 'yahoo' in data_path:
            vocab_file = os.path.join(data_path, "vocab.txt")
        vocab_dict = self.build_vocab(
            train_file, max_vocab_cnt, vocab_file=vocab_file)

        train_ids = self.corpus_to_token_ids(train_file, vocab_dict)

        valid_ids = self.corpus_to_token_ids(valid_file, vocab_dict)
        test_ids = self.corpus_to_token_ids(test_file, vocab_dict)

        return train_ids, valid_ids, test_ids

    def generate_dataset(self,
                         raw_data,
                         batch_size,
                         sort_cache=False,
                         cache_num=20,
                         mode='train'):
        src_data = raw_data
        in_tar = [data[:-1] for data in src_data]
        label_tar = [data[1:] for data in src_data]

        data_len = len(src_data)
        index = np.arange(data_len)
        if mode == "train":
            np.random.shuffle(index)

        def pad_batch(data):
            max_len = 0
            for ele in data:
                if len(ele) > max_len:
                    max_len = len(ele)

            ids = np.ones((len(data), max_len), dtype='int64') * self.PAD_ID
            mask = np.zeros((len(data)), dtype='int32')
            for i, ele in enumerate(data):
                ids[i, :len(ele)] = ele
                mask[i] = len(ele)
            return ids, mask

        data = []
        cache_src = []
        if mode != "train":
            cache_num = 1
        for j in range(data_len):
            # Generate cache_num batch, sort them according to sequence length
            if len(cache_src) == batch_size * cache_num:
                if sort_cache:
                    # Sort source suquences of a cache according to source sequence length
                    new_cache = sorted(cache_src, key=lambda k: len(k))
                else:
                    new_cache = cache_src
                for i in range(cache_num):
                    # Choose a batch
                    batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
                    # Construct trg and label according to src
                    trg = [data[:-1] for data in batch_data]
                    label = [data[1:] for data in batch_data]
                    # Pad for src, trg and label
                    src_ids, src_mask = pad_batch(batch_data)
                    trg, trg_mask = pad_batch(trg)
                    label, _ = pad_batch(label)
                    data.append((src_ids, src_mask, trg, trg_mask, label))
                cache_src = []
            cache_src.append(src_data[index[j]])

        if len(cache_src) > 0:
            if sort_cache:
                # Sort the rest of data according to src length
                new_cache = sorted(cache_src, key=lambda k: len(k))
            else:
                new_cache = cache_src

            for i in range(0, len(cache_src), batch_size):
                # Choose a batch
                end_index = min(i + batch_size, len(cache_src))
                batch_data = new_cache[i:end_index]
                # Construct trg and label according to src
                trg = [data[:-1] for data in batch_data]
                label = [data[1:] for data in batch_data]
                # Pad for src, trg and label
                src_ids, src_mask = pad_batch(batch_data)
                trg, trg_mask = pad_batch(trg)
                label, _ = pad_batch(label)
                data.append((src_ids, src_mask, trg, trg_mask, label))
        return data

    def __getitem__(self, index):
        src_ids, src_mask, in_tar, trg_mask, label_tar = self.data[index]
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1]))
        in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1]))
        label_tar = label_tar.reshape(
            (label_tar.shape[0], label_tar.shape[1], 1))
        return src_ids, src_mask, in_tar, trg_mask, label_tar

    def __len__(self):
        return len(self.data)


def create_data_loader(data_path,
                       batch_size,
                       max_seq_len=128,
                       sort_cache=False,
                       cache_num=20,
                       vocab_count_cn=-1,
                       PAD_ID=0,
                       BOS_ID=1,
                       EOS_ID=2,
                       UNK_ID=3,
                       mode='train'):
    train_dataset = VAEDataset(
        data_path,
        batch_size,
        sort_cache=sort_cache,
        cache_num=cache_num,
        max_vocab_cnt=-1,
        PAD_ID=0,
        BOS_ID=1,
        EOS_ID=2,
        UNK_ID=3,
        mode='train')
    valid_dataset = VAEDataset(
        data_path,
        batch_size,
        cache_num=cache_num,
        max_vocab_cnt=-1,
        PAD_ID=0,
        BOS_ID=1,
        EOS_ID=2,
        UNK_ID=3,
        mode='eval')
    test_dataset = VAEDataset(
        data_path,
        batch_size,
        cache_num=cache_num,
        max_vocab_cnt=-1,
        PAD_ID=0,
        BOS_ID=1,
        EOS_ID=2,
        UNK_ID=3,
        mode='test')

    # FIXME(liujiaqi06): set batch_size = None
    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=1, shuffle=True, drop_last=False)

    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda x: x[0],
        return_list=True)
    valid_loader = paddle.io.DataLoader(
        valid_dataset, return_list=True, collate_fn=lambda x: x[0])
    test_loader = paddle.io.DataLoader(
        test_dataset, return_list=True, collate_fn=lambda x: x[0])
    return train_loader, valid_loader, test_loader, train_dataset.train_data_len


def get_vocab(dataset, batch_size, vocab_file=None, max_sequence_len=50):
    train_dataset = VAEDataset(dataset, batch_size, mode='train')
    if dataset == 'yahoo':
        dataset_name = 'yahoo-answer-100k'
    else:
        dataset_name = os.path.join('simple-examples', 'data')

    dataset_prefix = os.path.join("data", dataset_name)

    train_file = os.path.join(dataset_prefix, dataset + ".train.txt")
    vocab_file = None
    if "yahoo" in dataset:
        vocab_file = os.path.join(dataset_prefix, "vocab.txt")
    src_vocab = train_dataset.build_vocab(train_file, vocab_file=vocab_file)
    rev_vocab = {}
    for key, value in src_vocab.items():
        rev_vocab[value] = key
    return rev_vocab, train_dataset.BOS_ID, train_dataset.EOS_ID
