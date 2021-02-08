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
#coding: utf-8
"""
The file_reader converts raw corpus to input.
"""
import os
import __future__
import io


def file_reader(file_dir,
                word2id_dict,
                label2id_dict,
                word_replace_dict,
                filename_feature=""):
    """
    define the reader to read files in file_dir
    """
    word_dict_len = max(map(int, word2id_dict.values())) + 1
    label_dict_len = max(map(int, label2id_dict.values())) + 1

    def reader():
        """
        the data generator
        """
        index = 0
        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                for line in io.open(
                        os.path.join(root, filename), 'r', encoding='utf8'):
                    index += 1
                    bad_line = False
                    line = line.strip("\n")
                    if len(line) == 0:
                        continue
                    seg_tag = line.rfind("\t")
                    word_part = line[0:seg_tag].strip().split(' ')
                    label_part = line[seg_tag + 1:]
                    word_idx = []
                    words = word_part
                    for word in words:
                        if word in word_replace_dict:
                            word = word_replace_dict[word]
                        if word in word2id_dict:
                            word_idx.append(int(word2id_dict[word]))
                        else:
                            word_idx.append(int(word2id_dict["<UNK>"]))
                    target_idx = []
                    labels = label_part.strip().split(" ")
                    for label in labels:
                        if label in label2id_dict:
                            target_idx.append(int(label2id_dict[label]))
                        else:
                            target_idx.append(int(label2id_dict["O"]))
                    if len(word_idx) != len(target_idx):
                        print(line)
                        continue
                    yield word_idx, target_idx

    return reader


def test_reader(file_dir,
                word2id_dict,
                label2id_dict,
                word_replace_dict,
                filename_feature=""):
    """
    define the reader to read test files in file_dir
    """
    word_dict_len = max(map(int, word2id_dict.values())) + 1
    label_dict_len = max(map(int, label2id_dict.values())) + 1

    def reader():
        """
        the data generator
        """
        index = 0
        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                if not filename.startswith(filename_feature):
                    continue
                for line in io.open(
                        os.path.join(root, filename), 'r', encoding='utf8'):
                    index += 1
                    bad_line = False
                    line = line.strip("\n")
                    if len(line) == 0:
                        continue
                    seg_tag = line.rfind("\t")
                    if seg_tag == -1:
                        seg_tag = len(line)
                    word_part = line[0:seg_tag]
                    label_part = line[seg_tag + 1:]
                    word_idx = []
                    words = word_part
                    for word in words:
                        if ord(word) < 0x20:
                            word = ' '
                        if word in word_replace_dict:
                            word = word_replace_dict[word]
                        if word in word2id_dict:
                            word_idx.append(int(word2id_dict[word]))
                        else:
                            word_idx.append(int(word2id_dict["OOV"]))
                    yield word_idx, words

    return reader


def load_reverse_dict(dict_path):
    """
    Load a dict. The first column is the key and the second column is the value.
    """
    result_dict = {}
    # TODO 字和词模型
    for idx, line in enumerate(io.open(dict_path, "r", encoding='utf8')):
        terms = line.strip("\n")
        result_dict[terms] = idx
    return result_dict


def load_dict(dict_path):
    """
    Load a dict. The first column is the value and the second column is the key.
    """
    result_dict = {}
    for idx, line in enumerate(io.open(dict_path, "r", encoding='utf8')):
        terms = line.strip("\n")
        result_dict[idx] = terms
    return result_dict
