#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
The file_reader converts raw corpus to input.
"""

import os

import numpy as np
import paddle
from paddlenlp.datasets import MapDataset

# We use "\002" to separate sentence characters and sequence labels,
# for example: 除\002了\002他\002续\002任\002十\002二\002届\002政\002协\002委\002员
#              p-B\002p-I\002r-B\002v-B\002v-I\002m-B\002m-I\002m-I\002ORG-B\002ORG-I\002n-B\002n-I\002
CHAR_DELIMITER = "\002"


def load_dataset(datafiles):

    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            if "infer" in data_path:
                next(fp)
            for line in fp:
                line = line.strip()
                if "infer" in data_path:
                    words = list(line)
                    yield [words]
                else:
                    words, labels = line.split("\t")
                    words = words.split(CHAR_DELIMITER)
                    labels = labels.split(CHAR_DELIMITER)
                    assert len(words) == len(
                        labels
                    ), "The word %s is not match with the label %s" % (words,
                                                                       labels)
                    yield [words, labels]

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def load_vocab(dict_path):
    """
    Load vocab from file
    """
    vocab = {}
    reverse = None
    with open(dict_path, "r", encoding='utf8') as fin:
        for i, line in enumerate(fin):
            terms = line.strip("\n").split("\t")
            if len(terms) == 2:
                if reverse == None:
                    reverse = True if terms[0].isdigit() else False
                if reverse:
                    value, key = terms
                else:
                    key, value = terms
            elif len(terms) == 1:
                key, value = terms[0], i
            else:
                raise ValueError("Error line: %s in file: %s" %
                                 (line, dict_path))
            vocab[key] = value
    return vocab


def normalize_token(token, normlize_vocab):
    """Normalize text from DBC case to SBC case"""
    if normlize_vocab:
        token = normlize_vocab.get(token, token)
    return token


def convert_tokens_to_ids(tokens,
                          vocab,
                          oov_replace_token=None,
                          normlize_vocab=None):
    """convert tokens to token indexs"""
    token_ids = []
    oov_replace_token = vocab.get(
        oov_replace_token) if oov_replace_token else None
    for token in tokens:
        token = normalize_token(token, normlize_vocab)
        token_id = vocab.get(token, oov_replace_token)
        token_ids.append(token_id)

    return token_ids


def convert_example(example,
                    max_seq_len,
                    word_vocab,
                    label_vocab=None,
                    normlize_vocab=None):
    if len(example) == 2:
        tokens, labels = example
    else:
        tokens, labels = example[0], None
    tokens = tokens[:max_seq_len]

    token_ids = convert_tokens_to_ids(tokens,
                                      word_vocab,
                                      oov_replace_token="OOV",
                                      normlize_vocab=normlize_vocab)
    length = len(token_ids)
    if labels is not None:
        labels = labels[:max_seq_len]
        label_ids = convert_tokens_to_ids(labels,
                                          label_vocab,
                                          oov_replace_token="O")
        return token_ids, length, label_ids
    else:
        return token_ids, length


def parse_result(words, preds, lengths, word_vocab, label_vocab):
    """ parse padding result """
    batch_out = []
    id2word_dict = dict(zip(word_vocab.values(), word_vocab.keys()))
    id2label_dict = dict(zip(label_vocab.values(), label_vocab.keys()))
    for sent_index in range(len(lengths)):
        sent = [
            id2word_dict[index]
            for index in words[sent_index][:lengths[sent_index]]
        ]
        tags = [
            id2label_dict[index]
            for index in preds[sent_index][:lengths[sent_index]]
        ]

        sent_out = []
        tags_out = []
        parital_word = ""
        for ind, tag in enumerate(tags):
            # for the first word
            if parital_word == "":
                parital_word = sent[ind]
                tags_out.append(tag.split('-')[0])
                continue

            # for the beginning of word
            if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
                sent_out.append(parital_word)
                tags_out.append(tag.split('-')[0])
                parital_word = sent[ind]
                continue

            parital_word += sent[ind]

        # append the last word, except for len(tags)=0
        if len(sent_out) < len(tags_out):
            sent_out.append(parital_word)

        batch_out.append([sent_out, tags_out])
    return batch_out
