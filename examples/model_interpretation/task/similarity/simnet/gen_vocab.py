#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#!/usr/bin/env python
# coding=utf-8

import sys

sys.path.append("../../..")
from paddlenlp.datasets import load_dataset
from collections import defaultdict
import spacy

if sys.argv[1] == 'ch':
    train_ds, dev_ds, test_ds = load_dataset("lcqmc",
                                             splits=["train", "dev", "test"])

    vocab = defaultdict(int)
    for example in train_ds.data:
        query = example['query']
        title = example['title']
        for c in query:
            vocab[c] += 1
        for c in title:
            vocab[c] += 1
    with open("vocab.char", "w") as f:
        for k, v in vocab.items():
            if v > 3:
                f.write(k + '\n')

else:
    tokenizer = spacy.load('en_core_web_sm')
    vocab = defaultdict(int)

    with open('../data/QQP/train/train.tsv', 'r') as f_dataset:
        for idx, line in enumerate(f_dataset.readlines()):
            if idx == 0:
                continue
            line_split = line.strip().split('\t')
            query = [token.text for token in tokenizer(line_split[0])]
            title = [token.text for token in tokenizer(line_split[1])]

            for word in query:
                vocab[word] += 1
            for word in title:
                vocab[word] += 1

    with open("vocab_QQP", "w") as f:
        for k, v in vocab.items():
            if v > 3:
                f.write(k + '\n')
