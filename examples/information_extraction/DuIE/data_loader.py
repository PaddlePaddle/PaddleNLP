# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
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

import collections
import json
import os
from typing import Optional, List, Union, Dict
from dataclasses import dataclass

import numpy as np
import paddle
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, PretrainedTokenizer
from paddlenlp.utils.log import logger

from extract_chinese_and_punct import ChineseAndPunctuationExtractor

InputFeature = collections.namedtuple("InputFeature", [
    "input_ids", "seq_len", "tok_to_orig_start_index", "tok_to_orig_end_index",
    "labels"
])


def parse_label(spo_list, label_map, tokens, tokenizer):
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    seq_len = len(tokens)
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    #  find all entities and tag them with corresponding "B"/"I" labels
    for spo in spo_list:
        for spo_object in spo['object'].keys():
            # assign relation label
            if spo['predicate'] in label_map.keys():
                # simple relation
                label_subject = label_map[spo['predicate']]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object']['@value'])
            else:
                # complex relation
                label_subject = label_map[spo['predicate'] + '_' + spo_object]
                label_object = label_subject + 55
                subject_tokens = tokenizer._tokenize(spo['subject'])
                object_tokens = tokenizer._tokenize(spo['object'][spo_object])

            subject_tokens_len = len(subject_tokens)
            object_tokens_len = len(object_tokens)

            # assign token label
            # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
            # to prevent single token from being labeled into two different entity
            # we tag the longer entity first, then match the shorter entity within the rest text
            forbidden_index = None
            if subject_tokens_len > object_tokens_len:
                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        labels[index][label_subject] = 1
                        for i in range(subject_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        if forbidden_index is None:
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        # check if labeled already
                        elif index < forbidden_index or index >= forbidden_index + len(
                                subject_tokens):
                            labels[index][label_object] = 1
                            for i in range(object_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

            else:
                for index in range(seq_len - object_tokens_len + 1):
                    if tokens[index:index + object_tokens_len] == object_tokens:
                        labels[index][label_object] = 1
                        for i in range(object_tokens_len - 1):
                            labels[index + i + 1][1] = 1
                        forbidden_index = index
                        break

                for index in range(seq_len - subject_tokens_len + 1):
                    if tokens[index:index +
                              subject_tokens_len] == subject_tokens:
                        if forbidden_index is None:
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break
                        elif index < forbidden_index or index >= forbidden_index + len(
                                object_tokens):
                            labels[index][label_subject] = 1
                            for i in range(subject_tokens_len - 1):
                                labels[index + i + 1][1] = 1
                            break

    # if token wasn't assigned as any "B"/"I" tag, give it an "O" tag for outside
    for i in range(seq_len):
        if labels[i] == [0] * num_labels:
            labels[i][0] = 1

    return labels


def convert_example_to_feature(
        example,
        tokenizer: PretrainedTokenizer,
        chineseandpunctuationextractor: ChineseAndPunctuationExtractor,
        label_map,
        max_length: Optional[int] = 512,
        pad_to_max_length: Optional[bool] = None):
    spo_list = example['spo_list'] if "spo_list" in example.keys() else None
    text_raw = example['text']

    sub_text = []
    buff = ""
    for char in text_raw:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
        else:
            buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    orig_to_tok_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        orig_to_tok_index.append(len(tokens))
        sub_tokens = tokenizer._tokenize(token)
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
            if len(tokens) >= max_length - 2:
                break
        else:
            continue
        break

    seq_len = len(tokens)
    # 2 tags for each predicate + I tag + O tag
    num_labels = 2 * (len(label_map.keys()) - 2) + 2
    # initialize tag
    labels = [[0] * num_labels for i in range(seq_len)]
    if spo_list is not None:
        labels = parse_label(spo_list, label_map, tokens, tokenizer)

    # add [CLS] and [SEP] token, they are tagged into "O" for outside
    if seq_len > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
        labels = labels[0:(max_length - 2)]
        tok_to_orig_start_index = tok_to_orig_start_index[0:(max_length - 2)]
        tok_to_orig_end_index = tok_to_orig_end_index[0:(max_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    # "O" tag for [PAD], [CLS], [SEP] token
    outside_label = [[1] + [0] * (num_labels - 1)]

    labels = outside_label + labels + outside_label
    tok_to_orig_start_index = [-1] + tok_to_orig_start_index + [-1]
    tok_to_orig_end_index = [-1] + tok_to_orig_end_index + [-1]
    if seq_len < max_length:
        tokens = tokens + ["[PAD]"] * (max_length - seq_len - 2)
        labels = labels + outside_label * (max_length - len(labels))
        tok_to_orig_start_index = tok_to_orig_start_index + [-1] * (
            max_length - len(tok_to_orig_start_index))
        tok_to_orig_end_index = tok_to_orig_end_index + [-1] * (
            max_length - len(tok_to_orig_end_index))

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return InputFeature(
        input_ids=np.array(token_ids),
        seq_len=np.array(seq_len),
        tok_to_orig_start_index=np.array(tok_to_orig_start_index),
        tok_to_orig_end_index=np.array(tok_to_orig_end_index),
        labels=np.array(labels),
    )


class DuIEDataset(paddle.io.Dataset):

    def __init__(self,
                 data,
                 label_map,
                 tokenizer,
                 max_length=512,
                 pad_to_max_length=False):
        super(DuIEDataset, self).__init__()

        self.data = data
        self.chn_punc_extractor = ChineseAndPunctuationExtractor()
        self.tokenizer = tokenizer
        self.max_seq_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        example = json.loads(self.data[item])
        input_feature = convert_example_to_feature(example, self.tokenizer,
                                                   self.chn_punc_extractor,
                                                   self.label_map,
                                                   self.max_seq_length,
                                                   self.pad_to_max_length)
        return {
            "input_ids":
            np.array(input_feature.input_ids, dtype="int64"),
            "seq_lens":
            np.array(input_feature.seq_len, dtype="int64"),
            "tok_to_orig_start_index":
            np.array(input_feature.tok_to_orig_start_index, dtype="int64"),
            "tok_to_orig_end_index":
            np.array(input_feature.tok_to_orig_end_index, dtype="int64"),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels":
            np.array(input_feature.labels, dtype="float32"),
        }

    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: PretrainedTokenizer,
                  max_length: Optional[int] = 512,
                  pad_to_max_length: Optional[bool] = None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(os.path.dirname(file_path),
                                      "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        with open(file_path, "r", encoding="utf-8") as fp:
            data = fp.readlines()
            return cls(data, label_map, tokenizer, max_length,
                       pad_to_max_length)


@dataclass
class DataCollator:
    """
    Collator for DuIE.
    """

    def __call__(self, examples: List[Dict[str, Union[list, np.ndarray]]]):
        batched_input_ids = np.stack([x['input_ids'] for x in examples])
        seq_lens = np.stack([x['seq_lens'] for x in examples])
        tok_to_orig_start_index = np.stack(
            [x['tok_to_orig_start_index'] for x in examples])
        tok_to_orig_end_index = np.stack(
            [x['tok_to_orig_end_index'] for x in examples])
        labels = np.stack([x['labels'] for x in examples])

        return (batched_input_ids, seq_lens, tok_to_orig_start_index,
                tok_to_orig_end_index, labels)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    d = DuIEDataset.from_file("./data/train_data.json", tokenizer)
    sampler = paddle.io.RandomSampler(data_source=d)
    batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=2)

    collator = DataCollator()
    loader = paddle.io.DataLoader(dataset=d,
                                  batch_sampler=batch_sampler,
                                  collate_fn=collator,
                                  return_list=True)
    for dd in loader():
        model_input = {
            "input_ids": dd[0],
            "seq_len": dd[1],
            "tok_to_orig_start_index": dd[2],
            "tok_to_orig_end_index": dd[3],
            "labels": dd[4]
        }
        print(model_input)
