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

from paddlenlp.transformers import BertModel, BertForPretraining, BertPretrainingCriterion
from paddlenlp.transformers import BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification

from common_test import CommonTest
from util import softmax_with_cross_entropy


def create_input_data(config):
    np.random.seed(102)
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
    del config['seq_len']
    del config['batch_size']

    return input_ids, masked_lm_positions


def create_bert_model(config, filename, TEST_MODEL_CLASS):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(dir_path, '{}.pdparams'.format(filename))
    if not os.path.exists(model_file):
        paddle.seed(102)
        bert = BertModel(**config)
        model = TEST_MODEL_CLASS(bert)
        paddle.save(model.state_dict(), model_file)
    return model_file


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
        self.config = BertModel.pretrained_init_configuration[
            'bert-base-uncased']
        self.config['num_hidden_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attention_probs_dropout_prob'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['intermediate_size'] = 1024
        self.config['seq_len'] = 64
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 512
        self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)
        self.model_file = create_bert_model(self.config, self.model_name,
                                            self.TEST_MODEL_CLASS)

    def set_output(self):
        self.expected_output = np.array(
            [[0.33695394, -0.18878141], [0.33130634, -0.08542377]])

    def set_model_file_name(self):
        self.model_name = "test_bert_cls"

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForSequenceClassification

    def setUp(self):
        self.set_model_file_name()
        self.set_model_class()
        self.set_input()
        self.set_output()

    def test_forward(self):
        bert = BertModel(**self.config)
        model = self.TEST_MODEL_CLASS(bert)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        output = model(input_ids)
        self.check_output_equal(output.numpy(), self.expected_output)


class TestBertForTokenClassification(TestBertForSequenceClassification):
    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForTokenClassification

    def set_output(self):
        self.expected_output = np.array([0.14911847])

    def test_forward(self):
        bert = BertModel(**self.config)
        model = self.TEST_MODEL_CLASS(bert)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        output = model(input_ids)
        self.check_output_equal(output.mean().numpy(), self.expected_output)


class TestBertForPretraining(TestBertForSequenceClassification):
    def set_model_file_name(self):
        self.model_name = "test_bert_pretrain"

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForPretraining

    def set_output(self):
        self.expected_prediction_scores_mean = np.array([0.00901102])
        self.expected_seq_relationship_score = np.array(
            [[-0.26995760, 0.52348071], [-0.13225232, 0.39070851]])

    def test_forward(self):
        bert = BertModel(**self.config)
        model = self.TEST_MODEL_CLASS(bert)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        masked_lm_positions = paddle.to_tensor(self.masked_lm_positions)
        output = model(input_ids, masked_positions=masked_lm_positions)

        prediction_scores_mean = output[0].mean().numpy()
        seq_relationship_score = output[1].numpy()
        self.check_output_equal(prediction_scores_mean,
                                self.expected_prediction_scores_mean)
        self.check_output_equal(seq_relationship_score,
                                self.expected_seq_relationship_score)


class TestBertForQuestionAnswering(TestBertForSequenceClassification):
    def set_model_class(self):
        self.TEST_MODEL_CLASS = BertForQuestionAnswering

    def set_output(self):
        self.expected_start_logit_mean = np.array([0.16729736])
        self.expected_end_logit_mean = np.array([0.13093957])

    def test_forward(self):
        bert = BertModel(**self.config)
        model = self.TEST_MODEL_CLASS(bert)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        output = model(input_ids)
        self.check_output_equal(output[0].mean().numpy(),
                                self.expected_start_logit_mean)
        self.check_output_equal(output[1].mean().numpy(),
                                self.expected_end_logit_mean)


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
