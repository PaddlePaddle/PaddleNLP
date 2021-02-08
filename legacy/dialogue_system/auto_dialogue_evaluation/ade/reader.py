# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"""Reader for auto dialogue evaluation"""

import io
import sys
import time
import random
import numpy as np
import os

import paddle
import paddle.fluid as fluid


class DataProcessor(object):
    def __init__(self, data_path, max_seq_length, batch_size):
        """init"""
        self.data_file = data_path
        self.max_seq_len = max_seq_length
        self.batch_size = batch_size
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_examples(self):
        """load examples"""
        examples = []
        index = 0
        fr = io.open(self.data_file, 'r', encoding="utf8")
        for line in fr:
            if index != 0 and index % 100 == 0:
                print("processing data: %d" % index)
            index += 1
            examples.append(line.strip())
        return examples

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")
        count = len(io.open(self.data_file, 'r', encoding="utf8").readlines())
        self.num_examples[phase] = count
        return self.num_examples[phase]

    def data_generator(self, place, phase="train", shuffle=True, sample_pro=1):
        """
        Generate data for train, dev or test.

        Args:
            phase: string. The phase for which to generate data.
            shuffle: bool. Whether to shuffle examples.
            sample_pro: sample data ratio
        """
        examples = self.get_examples()

        # used for ce
        if 'ce_mode' in os.environ:
            np.random.seed(0)
            random.seed(0)
            shuffle = False

        if shuffle:
            np.random.shuffle(examples)

        def batch_reader():
            """read batch data"""
            batch = []
            for example in examples:
                if sample_pro < 1:
                    if random.random() > sample_pro:
                        continue
                tokens = example.strip().split('\t')

                if len(tokens) != 3:
                    print("data format error: %s" % example.strip())
                    print("please input data: context \t response \t label")
                    continue

                context = [int(x) for x in tokens[0].split()[:self.max_seq_len]]
                response = [
                    int(x) for x in tokens[1].split()[:self.max_seq_len]
                ]
                label = [int(tokens[2])]
                instance = (context, response, label)

                if len(batch) < self.batch_size:
                    batch.append(instance)
                else:
                    if len(batch) == self.batch_size:
                        yield batch
                    batch = [instance]

            if len(batch) > 0:
                yield batch

        def create_lodtensor(data_ids, place):
            """create LodTensor for input ids"""
            cur_len = 0
            lod = [cur_len]
            seq_lens = [len(ids) for ids in data_ids]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            flattened_data = np.concatenate(data_ids, axis=0).astype("int64")
            flattened_data = flattened_data.reshape([len(flattened_data), 1])
            res = fluid.LoDTensor()
            res.set(flattened_data, place)
            res.set_lod([lod])
            return res

        def wrapper():
            """yield batch data to network"""
            for batch_data in batch_reader():
                context_ids = [batch[0] for batch in batch_data]
                response_ids = [batch[1] for batch in batch_data]
                label_ids = [batch[2] for batch in batch_data]
                context_res = create_lodtensor(context_ids, place)
                response_res = create_lodtensor(response_ids, place)
                label_ids = np.array(label_ids).astype("int64").reshape([-1, 1])
                input_batch = [context_res, response_res, label_ids]
                yield input_batch

        return wrapper
