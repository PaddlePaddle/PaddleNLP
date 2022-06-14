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
from paddlenlp.transformers import ElectraForMaskedLM, \
    ElectraForMultipleChoice, ElectraForQuestionAnswering , ElectraModel,ElectraForSequenceClassification

from common_test import CommonTest
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


class TestElectraForSequenceClassification(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            ElectraModel.pretrained_init_configuration['electra-base'])
        self.config['num_hidden_layers'] = 2
        self.config['vocab_size'] = 512
        self.config['attention_probs_dropout_prob'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['intermediate_size'] = 1024
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
        self.TEST_MODEL_CLASS = ElectraForSequenceClassification

    def check_testcase(self):
        self.check_output_equal(self.output.numpy().shape, self.expected_shape)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config['batch_size']
        del config['seq_len']

        bert = ElectraModel(**config)
        model = self.TEST_MODEL_CLASS(bert)
        input_ids = paddle.to_tensor(self.input_ids, dtype="int64")
        self.output = model(input_ids)
        self.check_testcase()


class TestElectraForMaskedLM(TestElectraForSequenceClassification):

    def set_model_class(self):
        self.TEST_MODEL_CLASS = ElectraForMaskedLM

    def set_output(self):
        self.expected_seq_shape = (self.config['batch_size'],
                                   self.config['seq_len'],
                                   self.config['vocab_size'])

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config['batch_size']
        del config['seq_len']

        electra = ElectraModel(**config)
        model = self.TEST_MODEL_CLASS(electra)
        input_ids = paddle.to_tensor(self.input_ids, dtype="int64")
        self.output = model(input_ids)
        self.check_testcase()

    def check_testcase(self):
        self.check_output_equal(self.output.numpy().shape,
                                self.expected_seq_shape)


class TestElectraForQuestionAnswering(TestElectraForSequenceClassification):

    def set_model_class(self):
        self.TEST_MODEL_CLASS = ElectraForQuestionAnswering

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


class TestElectraForMultipleChoice(TestElectraForSequenceClassification):

    def set_input(self):
        self.config = copy.deepcopy(
            ElectraModel.pretrained_init_configuration['electra-base'])
        self.config['num_hidden_layers'] = 2
        self.config['vocab_size'] = 512
        self.config['attention_probs_dropout_prob'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['intermediate_size'] = 1024
        self.config['seq_len'] = 64
        self.config['batch_size'] = 4
        self.config['num_choices'] = 2
        self.config['max_position_embeddings'] = 512

        self.input_ids = create_input_data(self.config)
        # [bs*num_choice,seq_l] -> [bs,num_choice,seq_l]
        self.input_ids = np.reshape(self.input_ids, [
            self.config['batch_size'] // self.config['num_choices'],
            self.config['num_choices'], -1
        ])

    def set_model_class(self):
        self.TEST_MODEL_CLASS = ElectraForMultipleChoice

    def set_output(self):
        self.expected_logit_shape = (self.config['batch_size'] //
                                     self.config['num_choices'],
                                     self.config['num_choices'])

    def check_testcase(self):
        self.check_output_equal(self.output.numpy().shape,
                                self.expected_logit_shape)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config["num_choices"]
        del config['batch_size']
        del config['seq_len']

        electra = ElectraModel(**config)
        model = self.TEST_MODEL_CLASS(electra)
        input_ids = paddle.to_tensor(self.input_ids, dtype="int64")
        self.output = model(input_ids)
        self.check_testcase()


if __name__ == "__main__":
    unittest.main()
