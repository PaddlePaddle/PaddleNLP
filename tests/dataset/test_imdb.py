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
from paddlenlp.datasets import Imdb

from common_test import CpuCommonTest
import util
import unittest


class TestImdbTrainSet(CpuCommonTest):
    def setUp(self):
        self.config['mode'] = 'train'
        np.random.seed(102)

    def test_training_set(self):
        expected_text, expected_label = (
            'Its a good movie maybe I like it because it was filmed here '
            'in PR The actors did a good performance and not only did the '
            'girls be girlish but they were good in fighting so it was awsome '
            'The guy is cute too so its a good match if you want to the guy '
            'or the girls', 1)
        expected_len = 25000

        train_ds = Imdb(**self.config)
        self.check_output_equal(len(train_ds), expected_len)
        self.check_output_equal(expected_text, train_ds[14][0])
        self.check_output_equal(expected_label, train_ds[14][1])


class TestImdbTestSet(CpuCommonTest):
    def setUp(self):
        self.config['mode'] = 'test'
        np.random.seed(102)

    def test_test_set(self):
        expected_text, expected_label = (
            'This is one of the great ones It works so beautifully that '
            'you hardly notice the miscasting of then 37 year old Dana '
            'Andrews as the drugstore soda jerk who goes to war and comes '
            'back four years later when he would have been at most 25 But '
            'then who else should have played him', 1)
        expected_len = 25000

        test_ds = Imdb(**self.config)
        self.check_output_equal(len(test_ds), expected_len)
        self.check_output_equal(expected_text, test_ds[2][0])
        self.check_output_equal(expected_label, test_ds[2][1])


class TestImdbWrongMode(CpuCommonTest):
    def setUp(self):
        # valid mode is 'train' and 'test', wrong mode would raise an error
        self.config['mode'] = 'wrong'

    @util.assert_raises
    def test_wrong_set(self):
        Imdb(**self.config)


if __name__ == "__main__":
    unittest.main()
