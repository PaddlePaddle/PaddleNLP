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

import os


def read_data(fname, word2idx):
    """
    Data is processed into a one-dimensional vector, and each value is the code corresponding to a word.
    The two sentences are separated by special characters < EOS >.
    
    Args:
        fname (str):
            data filename
        word2idx (dict):
            word dict
    
    Returns:
        list: return word vectors
    """
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise (Exception("[!] Data %s not found" % fname))

    words = []
    for line in lines:
        words.extend(line.split())

    print("Read %s words from %s" % (len(words), fname))

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])
    return data


def load_vocab(fname):
    """
    load word dict

    Args:
        fname (str): filename of the vocav file
    
    Returns:
        dict: word dict
    """
    word2idx = {}
    with open(fname, "r") as f:
        for line in f:
            pair = line.split()
            word2idx[pair[0]] = int(pair[1])
    return word2idx


def load_data(config):
    """
    load data
    
    Args:
        config: config
    
    Returns:
        word dict, and train, valid, test data
    """
    vocab_path = os.path.join(config.data_dir,
                              "%s.vocab.txt" % config.data_name)
    word2idx = load_vocab(vocab_path)

    train_data = read_data(
        os.path.join(config.data_dir, "%s.train.txt" % config.data_name),
        word2idx)
    valid_data = read_data(
        os.path.join(config.data_dir, "%s.valid.txt" % config.data_name),
        word2idx)
    test_data = read_data(
        os.path.join(config.data_dir, "%s.test.txt" % config.data_name),
        word2idx)

    return word2idx, train_data, valid_data, test_data
