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
        ('I had mixed feelings for Les Valseuses 1974 written and '
         'directed by Bertrand Blier when I started watching it but '
         'I ended up liking it I would not call it vulgar Dumb and'
         ' Dumber is vulgar The Sweetest Thing is both vulgar and '
         'unforgivably stupid I would call it shocking and offensive '
         'I can understand why many viewers especially the females '
         'would not like or even hate it It is the epitome of misogyny'
         ' or so it seems and the way two antiheroes treat every woman'
         ' theyd meet seems unspeakable But the more I think of it the '
         'more I realize that it somehow comes off as a delightful little '
         'gem I am fascinated how Blier was able to get away with it The '
         'movie is very entertaining and highly enjoyable it is well written'
         ' the acting by all is first  class and the music is sweet and '
         'melancholic Actually when I think of it two buddies had done something '
         'good to the women they came across to they prepared a woman in the train '
         'the lovely docile blonde Brigitte Fossey who started her movie career '
         'with one of the most impressive debuts in René Cléments Forbidden'
         ' Games1952 at age 6 for the meeting with her husband whom '
         'she had not seen for two months they found a man who was '
         'finally able to get a frigid MarieAnge MiouMiou exited and satisfied'
         ' they enlightened and educated young and very willing Isabelle Huppert '
         'in one of her early screen appearances Their encounter with Jeanne Moreau '
         'elevates this comedy to the tragic level In short I am not '
         'sure Id like to meet Gérard Depardieus JeanClaude and Patrick '
         'Dewaeres Pierrot in real life and invite them over for dinner '
         'but I had a good time watching the movie and two hours almost '
         'flew  it was never boring', 1),
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
    def set_config(self):
        self.config['path'] = 'imdb'
        self.config['splits'] = 'train'

    def test_train_set(self):
        expected_len = 25000
        expected_text, expected_label = get_examples(self.config['splits'])
        train_ds = load_dataset(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[0]['text'])
        self.check_output_equal(expected_label, train_ds[0]['label'])


class TestImdbTestSet(CpuCommonTest):
    def set_config(self):
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
    def set_config(self):
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

        self.check_output_equal(expected_train_text, ds[0][0]['text'])
        self.check_output_equal(expected_train_label, ds[0][0]['label'])
        self.check_output_equal(expected_test_text, ds[1][2]['text'])
        self.check_output_equal(expected_test_label, ds[1][2]['label'])


class TestImdbNoSplitDataFiles(CpuCommonTest):
    def set_config(self):
        self.config['path'] = 'imdb'

    def test_no_split_datafiles(self):
        with self.assertRaises(AssertionError):
            load_dataset(**self.config)
