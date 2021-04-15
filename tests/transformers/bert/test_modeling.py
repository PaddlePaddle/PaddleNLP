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

from paddlenlp.transformers import BertModel, BertForPretraining, BertPretrainingCriterion
from paddlenlp.transformers import BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification

from common_test import CommonTest
from util import softmax_with_cross_entropy, slow
import unittest


def create_input_data(config, seed=None):
    '''
    the generated input data will be same if a specified seed is set 
    '''
    if seed is not None:
        np.random.seed(seed)

    input_ids = np.random.randint(
        low=0,
        high=config['vocab_size'],
        size=(config["batch_size"], config["seq_len"]))
    num_to_predict = int(config["seq_len"] * 0.15)
    masked_lm_positions = np.random.choice(
        config["seq_len"], (config["batch_size"], num_to_predict),
        replace=False)
    masked_lm_positions = np.sort(masked_lm_positions)
    pred_padding_len = config["seq_len"] - num_to_predict
    temp_masked_lm_positions = np.full(
        masked_lm_positions.size, 0, dtype=np.int32)
    mask_token_num = 0
    for i, x in enumerate(masked_lm_positions):
        for j, pos in enumerate(x):
            temp_masked_lm_positions[mask_token_num] = i * config[
                "seq_len"] + pos
            mask_token_num += 1
    masked_lm_positions = temp_masked_lm_positions
    return input_ids, masked_lm_positions


class NpBertPretrainingCriterion(object):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, prediction_scores, seq_relationship_score,
                 masked_lm_labels, next_sentence_labels, masked_lm_scale):
        masked_lm_loss = softmax_with_cross_entropy(
            prediction_scores, masked_lm_labels, ignore_index=-1)
        masked_lm_loss = masked_lm_loss / masked_lm_scale
        next_sentence_loss = softmax_with_cross_entropy(seq_relationship_score,
                                                        next_sentence_labels)
        return np.sum(masked_lm_loss) + np.mean(next_sentence_loss)


class TestBertForSequenceClassification(CommonTest):
    def set_input(self):
        self.config = copy.deepcopy(BertModel.pretrained_init_configuration[
            'bert-base-uncased'])
        self.config['num_hidden_layers'] = 2
        self.config['vocab_size'] = 512
        self.config['attention_probs_dropout_prob'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['intermediate_size'] = 1024
        self.config['seq_len'] = 64
        self.config['batch_size'] = 3
        self.config['max_position_embeddings'] = 512
        self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)

    def set_output(self):
        self.expected_shape = (self.config['batch_size'], 2)

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForSequenceClassification

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

        bert = BertModel(**config)
        model = self.TEST_MODEL_CLASS(bert)
        input_ids = paddle.to_tensor(self.input_ids)
        self.output = model(input_ids)
        self.check_testcase()


class TestBertForTokenClassification(TestBertForSequenceClassification):
    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForTokenClassification

    def set_output(self):
        self.expected_shape = (self.config['batch_size'],
                               self.config['seq_len'], 2)


class TestBertForPretraining(TestBertForSequenceClassification):
    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForPretraining

    def set_output(self):
        self.expected_seq_shape = (self.masked_lm_positions.shape[0],
                                   self.config['vocab_size'])
        self.expected_pooled_shape = (self.config['batch_size'], 2)

    def test_forward(self):
        config = copy.deepcopy(self.config)
        del config['batch_size']
        del config['seq_len']

        bert = BertModel(**config)
        model = self.TEST_MODEL_CLASS(bert)
        input_ids = paddle.to_tensor(self.input_ids)
        masked_lm_positions = paddle.to_tensor(self.masked_lm_positions)
        self.output = model(input_ids, masked_positions=masked_lm_positions)
        self.check_testcase()

    def check_testcase(self):
        self.check_output_equal(self.output[0].numpy().shape,
                                self.expected_seq_shape)
        self.check_output_equal(self.output[1].numpy().shape,
                                self.expected_pooled_shape)


class TestBertForQuestionAnswering(TestBertForSequenceClassification):
    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForQuestionAnswering

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


class TestBertPretrainingCriterion(CommonTest):
    def setUp(self):
        self.config['vocab_size'] = 1024
        self.criterion = BertPretrainingCriterion(**self.config)
        self.np_criterion = NpBertPretrainingCriterion(**self.config)

    def _construct_input_data(self, mask_num, vocab_size, batch_size):
        prediction_scores = np.random.rand(
            mask_num, vocab_size).astype(paddle.get_default_dtype())
        seq_relationship_score = np.random.rand(
            batch_size, 2).astype(paddle.get_default_dtype())
        masked_lm_labels = np.random.randint(0, vocab_size, (mask_num, 1))
        next_sentence_labels = np.random.randint(0, 2, (batch_size, 1))
        masked_lm_scale = 1.0
        masked_lm_weights = np.random.randint(
            0, 2, (mask_num)).astype(paddle.get_default_dtype())
        return prediction_scores, seq_relationship_score, masked_lm_labels, \
            next_sentence_labels, masked_lm_scale, masked_lm_weights

    def test_forward(self):
        np_prediction_score, np_seq_relationship_score, np_masked_lm_labels, \
            np_next_sentence_labels, masked_lm_scale, np_masked_lm_weights \
            = self._construct_input_data(20, self.config['vocab_size'], 4)

        prediction_score = paddle.to_tensor(np_prediction_score)
        seq_relationship_score = paddle.to_tensor(np_seq_relationship_score)
        masked_lm_labels = paddle.to_tensor(np_masked_lm_labels)
        next_sentence_labels = paddle.to_tensor(np_next_sentence_labels)
        masked_lm_weights = paddle.to_tensor(np_masked_lm_weights)

        np_loss = self.np_criterion(
            np_prediction_score, np_seq_relationship_score, np_masked_lm_labels,
            np_next_sentence_labels, masked_lm_scale)
        loss = self.criterion(prediction_score, seq_relationship_score,
                              masked_lm_labels, next_sentence_labels,
                              masked_lm_scale)
        self.check_output_equal(np_loss, loss.numpy()[0])


class TestBertFromPretrain(CommonTest):
    @slow
    def test_bert_base_uncased(self):
        model = BertModel.from_pretrained(
            'bert-base-uncased',
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0)
        self.config = copy.deepcopy(model.config)
        self.config['seq_len'] = 32
        self.config['batch_size'] = 3

        input_ids, _ = create_input_data(self.config, 102)
        input_ids = paddle.to_tensor(input_ids)
        output = model(input_ids)

        expected_seq_shape = (self.config['batch_size'], self.config['seq_len'],
                              self.config['hidden_size'])
        expected_pooled_shape = (self.config['batch_size'],
                                 self.config['hidden_size'])
        self.check_output_equal(output[0].numpy().shape, expected_seq_shape)
        self.check_output_equal(output[1].numpy().shape, expected_pooled_shape)
        expected_seq_slice = np.array([[0.17383946, 0.09206937, 0.45788339],
                                       [-0.28287640, 0.06244858, 0.54864359],
                                       [-0.54589444, 0.04811822, 0.50559914]])
        # There's output diff about 1e-6 between cpu and gpu
        self.check_output_equal(
            output[0].numpy()[0, 0:3, 0:3], expected_seq_slice, atol=1e-6)

        expected_pooled_slice = np.array(
            [[-0.67418981, -0.07148759, 0.85799801],
             [-0.62072051, -0.08452632, 0.96691507],
             [-0.74019802, -0.10187808, 0.95353240]])
        self.check_output_equal(
            output[1].numpy()[0:3, 0:3], expected_pooled_slice, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
