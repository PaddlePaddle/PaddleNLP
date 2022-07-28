# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import copy
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import InputExample


def load_local_dataset(data_path, splits, label_list):
    """
    Load dataset for hierarchical classification from files, where
    there is one example per line. 
    Text and labels at different levels are seperated by '\t', and 
    multiple labels in the same level are delimited by ','.

    Args:
        data_path (str):
            Path to the dataset directory, including label.txt, train.txt, 
            dev.txt (and data.txt).
        splits (list):
            Which file(s) to load, such as ['train', 'dev', 'test'].
        label_list (dict):
            The dictionary that maps labels to indeces.
    """

    def _reader(data_file):
        with open(data_file, "r", encoding="utf-8") as fp:
            for idx, line in enumerate(fp):
                data = line.strip().split("\t")
                depth = len(data) - 1
                layers = [x.strip().split(",") for x in data[1:]]
                shape = [len(layer) for layer in layers]
                offsets = [0] * len(shape)
                has_next = True
                labels = []
                while has_next:
                    l = ''
                    for i, off in enumerate(offsets):
                        if l == '':
                            l = layers[i][off]
                        else:
                            l += '##{}'.format(layers[i][off])
                        if l not in labels:
                            labels.append(l)
                    for i in range(len(shape) - 1, -1, -1):
                        if offsets[i] + 1 >= shape[i]:
                            offsets[i] = 0
                            if i == 0:
                                has_next = False
                        else:
                            offsets[i] += 1
                            break
                yield InputExample(uid=idx,
                                   text_a=data[0],
                                   text_b=None,
                                   labels=labels)

    split_map = {"train": "train.txt", "dev": "dev.txt", "test": "data.txt"}
    datasets = []
    for split in splits:
        data_file = os.path.join(data_path, split_map[split])
        datasets.append(
            load_dataset(_reader,
                         data_file=data_file,
                         label_list=label_list,
                         lazy=False))
    return datasets


def convert_fn(labels_to_ids, example):
    """
    Self-defined function to create one-hot labels.
    """
    if isinstance(example, InputExample):
        wrapped = copy.deepcopy(example)
        wrapped.labels = [
            float(1) if x in wrapped.labels else float(0)
            for x in range(len(labels_to_ids))
        ]
        return wrapped
    else:
        raise TypeError('InputExample')
