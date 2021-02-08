#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
EmoTect utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import six
import random

import paddle
import paddle.fluid as fluid
import numpy as np


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """
    Init CheckPoint
    """

    fluid.load(main_program, init_checkpoint_path, exe)


def word2id(word_dict, query):
    """
    Convert word sequence into id list
    """
    unk_id = len(word_dict)
    wids = [
        word_dict[w] if w in word_dict else unk_id
        for w in query.strip().split(" ")
    ]
    return wids


def pad_wid(wids, max_seq_len=128, pad_id=0):
    """
    Padding data to max_seq_len
    """
    seq_len = len(wids)
    if seq_len < max_seq_len:
        for i in range(max_seq_len - seq_len):
            wids.append(pad_id)
    else:
        wids = wids[:max_seq_len]
        seq_len = max_seq_len
    return wids, seq_len


def data_reader(file_path, word_dict, num_examples, phrase, epoch, max_seq_len):
    """
    Data reader, which convert word sequence into id list
    """
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith("label"):
                continue
            if phrase == "infer":
                cols = line.strip().split("\t")
                query = cols[-1] if len(cols) != -1 else cols[0]
                wids = word2id(word_dict, query)
                wids, seq_len = pad_wid(wids, max_seq_len)
                all_data.append((wids, seq_len))
            else:
                cols = line.strip().split("\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = int(cols[0])
                query = cols[1].strip()
                wids = word2id(word_dict, query)
                wids, seq_len = pad_wid(wids, max_seq_len)
                all_data.append((wids, label, seq_len))
    num_examples[phrase] = len(all_data)

    if phrase == "infer":

        def reader():
            """
            Infer reader function
            """
            for wids, seq_len in all_data:
                yield wids, seq_len

        return reader

    def reader():
        """
        Reader function
        """
        for idx in range(epoch):
            if phrase == "train" and 'ce_mode' not in os.environ:
                random.shuffle(all_data)
            for wids, label, seq_len in all_data:
                yield wids, label, seq_len

    return reader


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as fin:
        wid = 0
        for line in fin:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def print_arguments(args):
    """
    print arguments
    """
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def query2ids(vocab_path, query):
    """
    Convert query to id list according to the given vocab
    """
    vocab = load_vocab(vocab_path)
    wids = word2id(vocab, query)
    return wids
