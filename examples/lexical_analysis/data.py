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

import paddle
import numpy as np

# We use "\002" to separate sentence characters and sequence labels,
# for example: 除\002了\002他\002续\002任\002十\002二\002届\002政\002协\002委\002员
#              p-B\002p-I\002r-B\002v-B\002v-I\002m-B\002m-I\002m-I\002ORG-B\002ORG-I\002n-B\002n-I\002
CHAR_DELIMITER = "\002"


class LacDataset(paddle.io.Dataset):
    """Load the dataset and convert all the texts to ids.

        Args:
            base_path (str): the path of the dataset directory.
            word_vocab (str): The path of the word dictionary.
            label_vocab (str): The path of the label dictionary.
            word_replace_dict (str): The path of the word replacement Dictionary.
            mode (str, optional): The load mode, "train", "test" or "infer". Defaults to 'train', meaning load the train dataset.
        """

    def __init__(self, base_path, mode='train'):
        self.mode = mode
        self.base_path = base_path
        word_dict_path = os.path.join(self.base_path, 'word.dic')
        label_dict_path = os.path.join(self.base_path, 'tag.dic')
        word_rep_dict_path = os.path.join(self.base_path, 'q2b.dic')
        self.word_vocab = self._load_vocab(word_dict_path)
        self.label_vocab = self._load_vocab(label_dict_path)
        self.word_replace_dict = self._load_vocab(word_rep_dict_path)

        # Calculate vocab size and labels number, note: vocab value strats from 0.
        self.vocab_size = len(self.word_vocab)
        self.num_labels = len(self.label_vocab)

        if self.mode in {"train", "test", "infer"}:
            self.dataset_path = os.path.join(self.base_path,
                                             "%s.tsv" % self.mode)
            self._read_file()
        else:
            raise ValueError(
                'Invalid mode: %s. Only support "train", "test" and "infer"' %
                self.mode)

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if self.mode == "infer":
            return [self.word_ids[index], len(self.word_ids[index])]
        else:
            return [
                self.word_ids[index], len(self.word_ids[index]),
                self.label_ids[index]
            ]

    def _read_file(self):
        self.word_ids = []
        self.label_ids = []
        self.total = 0
        with open(self.dataset_path, "r", encoding="utf-8") as fread:
            if self.mode != "infer":
                next(fread)
            for line in fread:
                line = line.strip()
                if self.mode == "infer":
                    words = list(line)
                else:
                    words, labels = line.split("\t")
                    words = words.split(CHAR_DELIMITER)

                tmp_word_ids = self._convert_tokens_to_ids(
                    words,
                    self.word_vocab,
                    oov_replace="OOV",
                    token_replace=self.word_replace_dict)

                self.word_ids.append(tmp_word_ids)
                if self.mode != "infer":
                    tmp_label_ids = self._convert_tokens_to_ids(
                        labels.split(CHAR_DELIMITER),
                        self.label_vocab,
                        oov_replace="O")
                    self.label_ids.append(tmp_label_ids)
                    assert len(tmp_word_ids) == len(
                        tmp_label_ids
                    ), "The word ids %s is not match with the label ids %s" % (
                        tmp_word_ids, tmp_label_ids)

                self.total += 1

    def _load_vocab(self, dict_path):
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

    def _convert_tokens_to_ids(self,
                               tokens,
                               vocab,
                               oov_replace=None,
                               token_replace=None):
        """convert tokens to token indexs"""
        token_ids = []
        oov_replace_token = vocab.get(oov_replace) if oov_replace else None
        for token in tokens:
            if token_replace:
                token = token_replace.get(token, token)
            token_id = vocab.get(token, oov_replace_token)
            token_ids.append(token_id)

        return token_ids


def parse_lac_result(words, preds, lengths, word_vocab, label_vocab):
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
