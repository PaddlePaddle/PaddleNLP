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
from paddlenlp.datasets import Imdb
from paddlenlp.utils.log import logger

from common_test import CpuCommonTest


class TestImdbTrainSet(CpuCommonTest):
    def setUp(self):
        self.config['mode'] = 'train'

    def test_training_set(self):
        np.random.seed(102)
        expected_text, expected_label = (
            'This is a strange cerebral surreal esoteric film If there is '\
            'such a thing as intellectual horror cinema this film is it I '\
            'started to get scared and wish there was someone else watching '\
            'it with me and it barely has a plot Im going to have to see this '\
            'film again multiple times before I feel I really understand it '\
            'If youre the kind of person who likes My Dinner With Andre and '\
            'films by Godard or if you do a lot of mindaltering drugs you '\
            'will probably enjoy this film Wow', 1)
        expected_len = 25000

        train_ds = Imdb(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[0][0])
        self.check_output_equal(expected_label, train_ds[0][1])


class TestImdbTestSet(CpuCommonTest):
    def setUp(self):
        self.config['mode'] = 'test'

    def test_test_set(self):
        np.random.seed(102)
        expected_text, expected_label = (
            'An extremely powerful film that certainly isnt appreciated enough '
            'Its impossible to describe the experience of watching it The '
            'recent UK television adaptation was shameful  too ordinary and '
            'bland This original manages to imprint itself in your memory', 1)
        expected_len = 25000

        test_ds = Imdb(**self.config)
        self.check_output_equal(len(test_ds), expected_len)
        self.check_output_equal(expected_text, test_ds[2][0])
        self.check_output_equal(expected_label, test_ds[2][1])


class TestImdbWrongMode(CpuCommonTest):
    def setUp(self):
        # valid mode is 'train' and 'test', wrong mode would raise an error
        self.config['mode'] = 'wrong'

    def test_wrong_set(self):
        with self.assertRaises(AssertionError):
            Imdb(**self.config)
