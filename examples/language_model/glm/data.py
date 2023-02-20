# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.datasets import load_dataset


def cnnmail_reader(data_file):
    def punctuation_standardization(string: str):
        punctuation_dict = {"\u201c": '"', "\u201d": '"', "\u2019": "'", "\u2018": "'", "\u2013": "-"}
        for key, value in punctuation_dict.items():
            string = string.replace(key, value)
        return string

    def detokenize(string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-", "[": "-LSB-", "]": "-RSB-", "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", '"')
        string = string.replace("``", '"')
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string

    source_texts, target_texts = [], []
    with open(f"{data_file}.source", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = punctuation_standardization(line)
            line = detokenize(line)
            source_texts.append(line)
    with open(f"{data_file}.target", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            line = punctuation_standardization(line)
            line = detokenize(line)
            target_texts.append(line)
    assert len(source_texts) == len(target_texts)
    example_list = []
    for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        if (idx + 1) % 20000 == 0:
            print(f"Processed {idx + 1} examples")
        source_text = "[sMASK] Content:" + source_text
        example = {"text_a": source_text, "text_b": target_text}
        example_list.append(example)
    return example_list


def load_local_dataset(data_path, splits):
    dataset = []
    for split in splits:
        data_file = os.path.join(data_path, split)
        dataset.append(load_dataset(cnnmail_reader, data_file=data_file, lazy=False))
    return dataset
