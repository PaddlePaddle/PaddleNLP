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
from paddlenlp.datasets import load_dataset

from common_test import CpuCommonTest
import util
import unittest


def get_examples(mode='train'):
    examples = {
        'train':
        ('I loved this movie since I was 7 and I saw it on the opening day '
         'It was so touching and beautiful I strongly recommend seeing for '
         'all Its a movie to watch with your family by farbr br My MPAA rating '
         'PG13 for thematic elements prolonged scenes of disastor nuditysexuality '
         'and some language', 1),
        'test':
        ('Felix in Hollywood is a great film The version I viewed was very well '
         'restored which is sometimes a problem with these silent era animated films '
         'It has some of Hollywoods most famous stars making cameo animated '
         'appearances A must for any silent film or animation enthusiast', 1)
    }
    return examples[mode]


class TestImdbTrainSet(CpuCommonTest):

    def setUp(self):
        self.config['path_or_read_func'] = 'imdb'
        self.config['splits'] = 'train'

    def test_train_set(self):
        expected_len = 25000
        expected_text, expected_label = get_examples(self.config['splits'])
        train_ds = load_dataset(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[36]['text'])
        self.check_output_equal(expected_label, train_ds[36]['label'])


class TestImdbTestSet(CpuCommonTest):

    def setUp(self):
        self.config['path_or_read_func'] = 'imdb'
        self.config['splits'] = 'test'

    def test_test_set(self):
        expected_len = 25000
        expected_text, expected_label = get_examples(self.config['splits'])
        test_ds = load_dataset(**self.config)
        self.check_output_equal(len(test_ds), expected_len)
        self.check_output_equal(expected_text, test_ds[23]['text'])
        self.check_output_equal(expected_label, test_ds[23]['label'])


class TestImdbTrainTestSet(CpuCommonTest):

    def setUp(self):
        self.config['path_or_read_func'] = 'imdb'
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

        self.check_output_equal(expected_train_text, ds[0][36]['text'])
        self.check_output_equal(expected_train_label, ds[0][36]['label'])
        self.check_output_equal(expected_test_text, ds[1][23]['text'])
        self.check_output_equal(expected_test_label, ds[1][23]['label'])


class TestImdbNoSplitDataFiles(CpuCommonTest):

    def setUp(self):
        self.config['path_or_read_func'] = 'imdb'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
