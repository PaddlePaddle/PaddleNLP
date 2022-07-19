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

import copy
import numpy as np
import paddle
from paddlenlp.transformers import GPTForSequenceClassification, GPTForTokenClassification, GPTModel
import random
from common_test import CommonTest
import unittest

from tests.transformers.electra.test_modeling import TestElectraForSequenceClassification


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


class TestGPTForSequenceClassification(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            GPTModel.pretrained_init_configuration['gpt2-medium-en'])
        self.config['num_hidden_layers'] = 2
        self.config['vocab_size'] = 512
        self.config['eos_token_id'] = 511
        self.config['attention_probs_dropout_prob'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['seq_len'] = 64
        self.config['batch_size'] = 4
        self.input_ids = create_input_data(self.config)

    def set_output(self):
        self.expected_shape = (self.config['batch_size'], 2)

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = GPTForSequenceClassification

    def check_testcase(self):
        self.check_output_equal(self.output.numpy().shape, self.expected_shape)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        input_ids = copy.deepcopy(self.input_ids)
        batch_size = config['batch_size']
        del config['batch_size']
        del config['seq_len']

        gpt = GPTModel(**config)
        model = self.TEST_MODEL_CLASS(gpt)
        # we need to test sentence with `eos_token_id`.
        for i in range(batch_size):
            index = random.randint(0, 10)
            input_ids[i, -index:] = self.config["eos_token_id"]
        input_ids = paddle.to_tensor(input_ids, dtype="int64")
        self.output = model(input_ids)
        self.check_testcase()


class TestGPTForTokenClassification(TestElectraForSequenceClassification):

    def set_model_class(self):
        self.TEST_MODEL_CLASS = GPTForTokenClassification

    def set_output(self):
        self.expected_shape = (self.config['batch_size'],
                               self.config['seq_len'], 2)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config['batch_size']
        del config['seq_len']

        gpt = GPTModel(**config)
        model = self.TEST_MODEL_CLASS(gpt)
        input_ids = paddle.to_tensor(self.input_ids, dtype="int64")
        self.output = model(input_ids)
        self.check_testcase()


if __name__ == "__main__":
    unittest.main()
