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
import copy

from paddlenlp.transformers import XLNetLMHeadModel, XLNetForMultipleChoice, XLNetForQuestionAnswering, XLNetModel

from common_test import CommonTest
from util import softmax_with_cross_entropy, slow
import unittest


def create_input_data(config, seed=None):
    '''
    the generated input data will be same if a specified seed is set
    '''
    if seed is not None:
        np.random.seed(seed)

    input_ids = np.random.randint(low=0,
                                  high=config['vocab_size'],
                                  size=(config["batch_size"],
                                        config["seq_len"]))

    return input_ids


class TestXLNetLMHeadModel(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            XLNetModel.pretrained_init_configuration['xlnet-base-cased'])
        self.config['n_layer'] = 2
        self.config['vocab_size'] = 512
        self.config['classifier_dropout'] = 0.0
        self.config['dropout'] = 0.0
        self.config['d_inner'] = 1024
        self.config['seq_len'] = 64
        self.config['batch_size'] = 4
        self.input_ids = create_input_data(self.config)

    def set_output(self):
        self.expected_shape = (self.config['batch_size'],
                               self.config['seq_len'], self.config['d_model'])

    def set_model_class(self):
        self.TEST_MODEL_CLASS = XLNetLMHeadModel

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def check_testcase(self):
        self.check_output_equal(self.output.numpy().shape, self.expected_shape)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config['batch_size']
        del config['seq_len']

        xlnet = XLNetModel(**config)
        model = self.TEST_MODEL_CLASS(xlnet)
        input_ids = paddle.to_tensor(self.input_ids)
        self.output = model(input_ids, return_dict=False)
        self.check_testcase()


class TestXLNetForMultipleChoice(TestXLNetLMHeadModel):

    def set_input(self):
        super(TestXLNetForMultipleChoice, self).set_input()
        self.config['num_choices'] = 2
        self.input_ids = np.reshape(self.input_ids, [
            self.config['batch_size'] // self.config['num_choices'],
            self.config['num_choices'], -1
        ])

    def set_model_class(self):
        self.TEST_MODEL_CLASS = XLNetForMultipleChoice

    def set_output(self):
        self.expected_logit_shape = (self.config['batch_size'] //
                                     self.config['num_choices'],
                                     self.config['num_choices'])

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config["num_choices"]
        del config['batch_size']
        del config['seq_len']

        xlnet = XLNetModel(**config)
        model = self.TEST_MODEL_CLASS(xlnet)
        input_ids = paddle.to_tensor(self.input_ids)
        self.output = model(input_ids, return_dict=False)
        self.check_testcase()


class TestXLNetForQuestionAnswering(TestXLNetLMHeadModel):

    def set_model_class(self):
        self.TEST_MODEL_CLASS = XLNetForQuestionAnswering

    def set_output(self):
        self.expected_start_logit_shape = (self.config['batch_size'],
                                           self.config['seq_len'])
        self.expected_end_logit_shape = (self.config['batch_size'],
                                         self.config['seq_len'])

    def check_testcase(self):
        self.check_output_equal(self.output[0].numpy().shape,
                                self.expected_start_logit_shape)
        self.check_output_equal(self.output[1].numpy().shape,
                                self.expected_end_logit_shape)


if __name__ == "__main__":
    unittest.main()
