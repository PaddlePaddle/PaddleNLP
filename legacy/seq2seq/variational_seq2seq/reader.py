# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import io
from collections import Counter
Py3 = sys.version_info[0] == 3

if Py3:
    line_tok = '\n'
else:
    line_tok = u'\n'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def _read_words(filename):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        if Py3:
            return f.read().replace("\n", "<EOS>").split()
        else:
            return f.read().decode("utf-8").replace(u"\n", u"<EOS>").split()


def read_all_line(filename):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def _vocab(vocab_file, train_file, max_vocab_cnt):
    lines = read_all_line(train_file)

    all_words = []
    for line in lines:
        all_words.extend(line.split())
    vocab_count = Counter(all_words).most_common()
    raw_vocab_size = min(len(vocab_count), max_vocab_cnt)
    with io.open(vocab_file, "w", encoding='utf-8') as f:
        for voc, fre in vocab_count[0:max_vocab_cnt]:
            f.write(voc)
            f.write(line_tok)


def _build_vocab(vocab_file, train_file=None, max_vocab_cnt=-1):
    if not os.path.exists(vocab_file):
        _vocab(vocab_file, train_file, max_vocab_cnt)
    vocab_dict = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    ids = 4
    with io.open(vocab_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            vocab_dict[line.strip()] = ids
            ids += 1
    # rev_vocab = {value:key for key, value in vocab_dict.items()}
    print("vocab word num", ids)

    return vocab_dict


def _para_file_to_ids(src_file, src_vocab):

    src_data = []
    with io.open(src_file, "r", encoding='utf-8') as f_src:
        for line in f_src.readlines():
            arra = line.strip().split()
            ids = [BOS_ID]
            ids.extend(
                [src_vocab[w] if w in src_vocab else UNK_ID for w in arra])
            ids.append(EOS_ID)
            src_data.append(ids)

    return src_data


def filter_len(src, max_sequence_len=128):
    new_src = []

    for id1 in src:
        if len(id1) > max_sequence_len:
            id1 = id1[:max_sequence_len]

        new_src.append(id1)

    return new_src


def raw_data(dataset_prefix, max_sequence_len=50, max_vocab_cnt=-1):

    src_vocab_file = dataset_prefix + ".vocab.txt"
    src_train_file = dataset_prefix + ".train.txt"
    src_eval_file = dataset_prefix + ".valid.txt"
    src_test_file = dataset_prefix + ".test.txt"

    src_vocab = _build_vocab(src_vocab_file, src_train_file, max_vocab_cnt)

    train_src = _para_file_to_ids(src_train_file, src_vocab)
    train_src = filter_len(train_src, max_sequence_len=max_sequence_len)
    eval_src = _para_file_to_ids(src_eval_file, src_vocab)

    test_src = _para_file_to_ids(src_test_file, src_vocab)

    return train_src, eval_src, test_src, src_vocab


def get_vocab(dataset_prefix, max_sequence_len=50):
    src_vocab_file = dataset_prefix + ".vocab.txt"
    src_vocab = _build_vocab(src_vocab_file)
    rev_vocab = {}
    for key, value in src_vocab.items():
        rev_vocab[value] = key

    return src_vocab, rev_vocab


def raw_mono_data(vocab_file, file_path):

    src_vocab = _build_vocab(vocab_file)
    test_src, test_tar = _para_file_to_ids( file_path, file_path, \
                                              src_vocab, src_vocab )

    return (test_src, test_tar)


def get_data_iter(raw_data,
                  batch_size,
                  sort_cache=False,
                  cache_num=1,
                  mode='train',
                  enable_ce=False):

    src_data = raw_data

    data_len = len(src_data)

    index = np.arange(data_len)
    if mode == "train" and not enable_ce:
        np.random.shuffle(index)

    def to_pad_np(data):
        max_len = 0
        for ele in data:
            if len(ele) > max_len:
                max_len = len(ele)

        ids = np.ones(
            (batch_size, max_len), dtype='int64') * PAD_ID  # PAD_ID = 0
        mask = np.zeros((batch_size), dtype='int32')

        for i, ele in enumerate(data):
            ids[i, :len(ele)] = ele
            mask[i] = len(ele)

        return ids, mask

    b_src = []

    if mode != "train":
        cache_num = 1
    for j in range(data_len):
        if len(b_src) == batch_size * cache_num:
            if sort_cache:
                new_cache = sorted(b_src, key=lambda k: len(k))
            new_cache = b_src
            for i in range(cache_num):
                batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
                src_ids, src_mask = to_pad_np(batch_data)
                yield (src_ids, src_mask)

            b_src = []

        b_src.append(src_data[index[j]])

    if len(b_src) > 0:
        if sort_cache:
            new_cache = sorted(b_src, key=lambda k: len(k))
        new_cache = b_src
        for i in range(0, len(b_src), batch_size):
            end_index = min((i + 1) * batch_size, len(b_src))
            batch_data = new_cache[i * batch_size:end_index]
            src_ids, src_mask = to_pad_np(batch_data)
            yield (src_ids, src_mask)
