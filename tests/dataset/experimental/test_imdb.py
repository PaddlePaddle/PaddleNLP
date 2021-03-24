# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import os
import unittest
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.utils.log import logger

from common_test import CpuCommonTest


def get_examples(mode='train'):
    examples = {
        'train':
        ('GREAT movie and the family will love it If kids are bored one day just '
         'pop the tape in and youll be so glad you didbr br Rubebr br i luv ravens',
         1),
        'test':
        ('I have loved this movie since I saw it in the theater in 1991 I was '
         '12 then and Wil Wheaton was my favorite actor and adolescent crush I '
         'am now 23 and I still love this movie The best part about it is '
         'whoever I am dating loves it too because it is a total machoguy movie '
         'It is wrought with enough action and mayhem to keep men with the '
         'shortest attention spans glued to the screen I only wish that it was '
         'available on DVD', 1)
    }
    return examples[mode]


class TestImdbTrainSet(CpuCommonTest):
    def setUp(self):
        self.config['path'] = 'imdb'
        self.config['splits'] = 'train'

    def test_train_set(self):
        expected_len = 25000
        expected_text, expected_label = get_examples(self.config['splits'])
        train_ds = load_dataset(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[2]['text'])
        self.check_output_equal(expected_label, train_ds[2]['label'])


class TestImdbTestSet(CpuCommonTest):
    def setUp(self):
        self.config['path'] = 'imdb'
        self.config['splits'] = 'test'

    def test_test_set(self):
        expected_len = 25000
        expected_text, expected_label = get_examples(self.config['splits'])
        test_ds = load_dataset(**self.config)
        self.check_output_equal(len(test_ds), expected_len)
        self.check_output_equal(expected_text, test_ds[2]['text'])
        self.check_output_equal(expected_label, test_ds[2]['label'])


class TestImdbTrainTestSet(CpuCommonTest):
    def setUp(self):
        self.config['path'] = 'imdb'
        self.config['splits'] = ['train', 'test']

    def test_train_set(self):
        expected_ds_num = 2
        expected_len = 25000
        expected_train_text, expected_train_label = get_examples('train')
        expected_test_text, expected_test_label = get_examples('test')
        ds = load_dataset(**self.config)

        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)
        self.check_output_equal(len(ds[1]), expected_len)

        self.check_output_equal(expected_train_text, ds[0][2]['text'])
        self.check_output_equal(expected_train_label, ds[0][2]['label'])
        self.check_output_equal(expected_test_text, ds[1][2]['text'])
        self.check_output_equal(expected_test_label, ds[1][2]['label'])


class TestImdbNoSplitDataFiles(CpuCommonTest):
    def setUp(self):
        self.config['path'] = 'imdb'

    def test_no_split_datafiles(self):
        with self.assertRaises(AssertionError):
            load_dataset(**self.config)
