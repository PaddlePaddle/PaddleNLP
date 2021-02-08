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
The file_reader converts raw corpus to input.
"""

import os
import argparse
import __future__
import io
import glob


def load_kv_dict(dict_path,
                 reverse=False,
                 delimiter="\t",
                 key_func=None,
                 value_func=None):
    """
    Load key-value dict from file
    """
    result_dict = {}
    for line in io.open(dict_path, "r", encoding='utf8'):
        terms = line.strip("\n").split(delimiter)
        if len(terms) != 2:
            continue
        if reverse:
            value, key = terms
        else:
            key, value = terms
        if key in result_dict:
            raise KeyError("key duplicated with [%s]" % (key))
        if key_func:
            key = key_func(key)
        if value_func:
            value = value_func(value)
        result_dict[key] = value
    return result_dict


class Dataset(object):
    """data reader"""

    def __init__(self, args, mode="train"):
        # read dict
        self.word2id_dict = load_kv_dict(
            args.word_dict_path, reverse=True, value_func=int)
        self.id2word_dict = load_kv_dict(args.word_dict_path)
        self.label2id_dict = load_kv_dict(
            args.label_dict_path, reverse=True, value_func=int)
        self.id2label_dict = load_kv_dict(args.label_dict_path)
        self.word_replace_dict = load_kv_dict(args.word_rep_dict_path)

    @property
    def vocab_size(self):
        """vocabuary size"""
        return max(self.word2id_dict.values()) + 1

    @property
    def num_labels(self):
        """num_labels"""
        return max(self.label2id_dict.values()) + 1

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in io.open(filename, "r", encoding='utf8'))

    def word_to_ids(self, words):
        """convert word to word index"""
        word_ids = []
        for word in words:
            word = self.word_replace_dict.get(word, word)
            if word not in self.word2id_dict:
                word = "OOV"
            word_id = self.word2id_dict[word]
            word_ids.append(word_id)

        return word_ids

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def file_reader(self, filename, max_seq_len=64, mode="train"):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """

        def wrapper():
            fread = io.open(filename, "r", encoding="utf-8")
            if mode == "infer":
                for line in fread:
                    words = line.strip()
                    word_ids = self.word_to_ids(words)
                    yield (word_ids[0:max_seq_len], )
            else:
                headline = next(fread)
                headline = headline.strip().split('\t')
                assert len(headline) == 2 and headline[
                    0] == "text_a" and headline[1] == "label"
                for line in fread:
                    words, labels = line.strip("\n").split("\t")
                    if len(words) < 1:
                        continue
                    word_ids = self.word_to_ids(words.split("\002"))
                    label_ids = self.label_to_ids(labels.split("\002"))
                    assert len(word_ids) == len(label_ids)
                    yield word_ids[0:max_seq_len], label_ids[0:max_seq_len]
            fread.close()

        return wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--word_dict_path",
        type=str,
        default="./conf/word.dic",
        help="word dict")
    parser.add_argument(
        "--label_dict_path",
        type=str,
        default="./conf/tag.dic",
        help="label dict")
    parser.add_argument(
        "--word_rep_dict_path",
        type=str,
        default="./conf/q2b.dic",
        help="word replace dict")
    args = parser.parse_args()
    dataset = Dataset(args)
    data_generator = dataset.file_reader("data/train.tsv")
    for word_idx, target_idx in data_generator():
        print(word_idx, target_idx)
        print(len(word_idx), len(target_idx))
        break
