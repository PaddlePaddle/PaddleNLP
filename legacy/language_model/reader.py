#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np

EOS = "</eos>"


def build_vocab(filename):

    vocab_dict = {}
    ids = 0
    vocab_dict[EOS] = ids
    ids += 1

    with open(filename, "r") as f:
        for line in f.readlines():
            for w in line.strip().split():
                if w not in vocab_dict:
                    vocab_dict[w] = ids
                    ids += 1

    print("vocab word num", ids)

    return vocab_dict


def file_to_ids(src_file, src_vocab):

    src_data = []
    with open(src_file, "r") as f_src:
        for line in f_src.readlines():
            arra = line.strip().split()
            ids = [src_vocab[w] for w in arra if w in src_vocab]

            src_data += ids + [0]
    return src_data


def get_ptb_data(data_path=None):

    train_file = os.path.join(data_path, "ptb.train.txt")
    valid_file = os.path.join(data_path, "ptb.valid.txt")
    test_file = os.path.join(data_path, "ptb.test.txt")

    vocab_dict = build_vocab(train_file)
    train_ids = file_to_ids(train_file, vocab_dict)
    valid_ids = file_to_ids(valid_file, vocab_dict)
    test_ids = file_to_ids(test_file, vocab_dict)

    return train_ids, valid_ids, test_ids


def get_data_iter(raw_data, batch_size, num_steps):
    data_len = len(raw_data)
    raw_data = np.asarray(raw_data, dtype="int64")

    batch_len = data_len // batch_size

    data = raw_data[0:batch_size * batch_len].reshape((batch_size, batch_len))

    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        start = i * num_steps
        x = np.copy(data[:, i * num_steps:(i + 1) * num_steps])
        y = np.copy(data[:, i * num_steps + 1:(i + 1) * num_steps + 1])

        yield (x, y)
