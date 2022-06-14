# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import json
from tqdm import tqdm


def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict((v, k) for k, v in word2id.items())

        return word2id, id2word


def read(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            items = line.strip().split("\t")
            assert len(items) == 3
            example = {
                "label": int(items[0]),
                "aspect_text": items[1],
                "text": items[2]
            }

            yield example


def convert_example_to_feature(example,
                               tokenizer,
                               label2id,
                               max_seq_len=512,
                               is_test=False):
    encoded_inputs = tokenizer(example["aspect_text"],
                               text_pair=example["text"],
                               max_seq_len=max_seq_len,
                               return_length=True)

    if not is_test:
        label = example["label"]
        return encoded_inputs["input_ids"], encoded_inputs[
            "token_type_ids"], encoded_inputs["seq_len"], label

    return encoded_inputs["input_ids"], encoded_inputs[
        "token_type_ids"], encoded_inputs["seq_len"]
