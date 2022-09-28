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
import unittest

import numpy as np
import paddle

from common_test import CommonTest
from util import softmax_with_cross_entropy, slow
from paddlenlp.transformers import BigBirdForSequenceClassification, \
    BigBirdPretrainingCriterion, BigBirdForPretraining, BigBirdModel, \
    BigBirdForQuestionAnswering, BigBirdForTokenClassification, BigBirdForMultipleChoice, \
    BigBirdForMaskedLM, BigBirdForCausalLM
from paddlenlp.transformers import create_bigbird_rand_mask_idx_list


def create_input_data(config, seed=None):
    if seed is not None:
        np.random.seed(seed)
    rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
        config["num_layers"], config["seq_len"], config["seq_len"],
        config["nhead"], config["block_size"], config["window_size"],
        config["num_global_blocks"], config["num_rand_blocks"], config["seed"])
    input_ids = np.random.randint(low=0,
                                  high=config['vocab_size'],
                                  size=(config["batch_size"],
                                        config["seq_len"]))
    num_to_predict = int(config["seq_len"] * 0.15)
    masked_lm_positions = np.random.choice(
        config["seq_len"], (config["batch_size"], num_to_predict),
        replace=False)
    masked_lm_positions = np.sort(masked_lm_positions)
    pred_padding_len = config["seq_len"] - num_to_predict
    temp_masked_lm_positions = np.full(masked_lm_positions.size,
                                       0,
                                       dtype=np.int32)
    mask_token_num = 0
    for i, x in enumerate(masked_lm_positions):
        for j, pos in enumerate(x):
            temp_masked_lm_positions[
                mask_token_num] = i * config["seq_len"] + pos
            mask_token_num += 1
    masked_lm_positions = temp_masked_lm_positions
    return rand_mask_idx_list, input_ids, masked_lm_positions


class NpBigBirdPretrainingCriterion(object):

    def __init__(self, vocab_size, use_nsp=False, ignore_index=0):
        self.vocab_size = vocab_size
        self.use_nsp = use_nsp
        self.ignore_index = ignore_index

    def __call__(self, prediction_scores, seq_relationship_score,
                 masked_lm_labels, next_sentence_labels, masked_lm_scale,
                 masked_lm_weights):
        masked_lm_loss = softmax_with_cross_entropy(
            prediction_scores, masked_lm_labels, ignore_index=self.ignore_index)
        masked_lm_loss = np.transpose(masked_lm_loss, [1, 0])
        masked_lm_loss = np.sum(masked_lm_loss * masked_lm_weights) / (
            np.sum(masked_lm_weights) + 1e-5)
        scale = 1.0
        if not self.use_nsp:
            scale = 0.0
        next_sentence_loss = softmax_with_cross_entropy(seq_relationship_score,
                                                        next_sentence_labels)
        return masked_lm_loss + np.mean(next_sentence_loss) * scale


class TestBigBirdForSequenceClassification(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)

    def set_output(self):
        self.expected_shape = (self.config['batch_size'], 2)

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForSequenceClassification

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        self.check_output_equal(self.expected_shape, output.numpy().shape)


class TestBigBirdForQuestionAnswering(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)

    def set_output(self):
        self.expected_shape1 = (self.config['batch_size'],
                                self.config['seq_len'])
        self.expected_shape2 = (self.config['batch_size'],
                                self.config['seq_len'])

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForQuestionAnswering

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        start_logits, end_logits = model(input_ids,
                                         rand_mask_idx_list=rand_mask_idx_list)
        self.check_output_equal(self.expected_shape1,
                                start_logits.numpy().shape)
        self.check_output_equal(self.expected_shape2, end_logits.numpy().shape)


class TestBigBirdForTokenClassification(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)
        self.num_classes = 2

    def set_output(self):
        self.expected_shape = (self.config['batch_size'],
                               self.config['seq_len'], self.num_classes)

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForTokenClassification

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird, num_classes=self.num_classes)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        self.check_output_equal(self.expected_shape, output.numpy().shape)


class TestBigBirdForMultipleChoice(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = [], [], []
        self.num_choices = 2
        for i in range(self.num_choices):
            rand_mask_idx_list, input_ids, masked_lm_positions = create_input_data(
                self.config)
            self.rand_mask_idx_list.append(rand_mask_idx_list)
            self.input_ids.append(input_ids)
            self.masked_lm_positions.append(masked_lm_positions)
        self.rand_mask_idx_list = np.array(self.rand_mask_idx_list).swapaxes(
            0, 1)
        self.input_ids = np.array(self.input_ids).swapaxes(0, 1)
        self.masked_lm_positions = np.array(self.masked_lm_positions).swapaxes(
            0, 1)

    def set_output(self):
        self.expected_shape = (self.config['batch_size'], self.num_choices)

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForMultipleChoice

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird, num_choices=self.num_choices)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        self.check_output_equal(self.expected_shape, output.numpy().shape)


class TestBigBirdForMaskedLM(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)
        self.labels = np.random.randint(low=0,
                                        high=self.config['vocab_size'],
                                        size=(self.config["batch_size"],
                                              self.config["seq_len"]))

    def set_output(self):
        self.expected_shape1 = (1, )
        self.expected_shape2 = (self.config['batch_size'],
                                self.config['seq_len'],
                                self.config['vocab_size'])
        self.expected_shape3 = (self.config['batch_size'],
                                self.config['seq_len'],
                                self.config['hidden_size'])

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForMaskedLM

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        labels = paddle.to_tensor(self.labels)
        masked_lm_loss, prediction_scores, sequence_output = model(
            input_ids, rand_mask_idx_list=rand_mask_idx_list, labels=labels)
        self.check_output_equal(self.expected_shape1,
                                masked_lm_loss.numpy().shape)
        self.check_output_equal(self.expected_shape2,
                                prediction_scores.numpy().shape)
        self.check_output_equal(self.expected_shape3,
                                sequence_output.numpy().shape)


class TestBigBirdForCausalLM(CommonTest):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 1024
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)
        self.labels = np.random.randint(low=0,
                                        high=self.config['vocab_size'],
                                        size=(self.config["batch_size"],
                                              self.config["seq_len"]))

    def set_output(self):
        self.expected_shape1 = (1, )
        self.expected_shape2 = (self.config['batch_size'],
                                self.config['seq_len'],
                                self.config['vocab_size'])
        self.expected_shape3 = (self.config['batch_size'],
                                self.config['seq_len'],
                                self.config['hidden_size'])

    def setUp(self):
        self.set_model_class()
        self.set_input()
        self.set_output()

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForCausalLM

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        labels = paddle.to_tensor(self.labels)
        masked_lm_loss, prediction_scores, sequence_output = model(
            input_ids, rand_mask_idx_list=rand_mask_idx_list, labels=labels)
        self.check_output_equal(self.expected_shape1,
                                masked_lm_loss.numpy().shape)
        self.check_output_equal(self.expected_shape2,
                                prediction_scores.numpy().shape)
        self.check_output_equal(self.expected_shape3,
                                sequence_output.numpy().shape)


class TestBigBirdForPretraining(TestBigBirdForSequenceClassification):

    def set_input(self):
        self.config = copy.deepcopy(
            BigBirdModel.pretrained_init_configuration['bigbird-base-uncased'])
        self.config['num_layers'] = 2
        self.config['vocab_size'] = 512
        self.config['attn_dropout'] = 0.0
        self.config['hidden_dropout_prob'] = 0.0
        self.config['dim_feedforward'] = 1024
        self.config['seq_len'] = 1024
        self.config['batch_size'] = 2
        self.config['max_position_embeddings'] = 2048
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)

    def set_model_class(self):
        self.TEST_MODEL_CLASS = BigBirdForPretraining

    def set_output(self):
        self.expected_pred_shape = (self.masked_lm_positions.shape[0],
                                    self.config['vocab_size'])
        self.expected_seq_shape = (self.config['batch_size'], 2)

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = self.TEST_MODEL_CLASS(bigbird)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        masked_positions = paddle.to_tensor(self.masked_lm_positions)
        output = model(input_ids,
                       rand_mask_idx_list=rand_mask_idx_list,
                       masked_positions=masked_positions)
        self.check_output_equal(output[0].numpy().shape,
                                self.expected_pred_shape)
        self.check_output_equal(output[1].numpy().shape,
                                self.expected_seq_shape)


class TestBigBirdPretrainingCriterionUseNSP(CommonTest):

    def setUp(self):
        self.config['vocab_size'] = 1024
        self.criterion = BigBirdPretrainingCriterion(**self.config)
        self.np_criterion = NpBigBirdPretrainingCriterion(**self.config)

    def _construct_input_data(self, mask_num, vocab_size, batch_size):
        prediction_scores = np.random.rand(mask_num, vocab_size).astype(
            paddle.get_default_dtype())
        seq_relationship_score = np.random.rand(batch_size, 2).astype(
            paddle.get_default_dtype())
        masked_lm_labels = np.random.randint(0, vocab_size, (mask_num, 1))
        next_sentence_labels = np.random.randint(0, 2, (batch_size, 1))
        masked_lm_scale = 1.0
        masked_lm_weights = np.random.randint(0, 2, (mask_num)).astype(
            paddle.get_default_dtype())
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

        np_loss = self.np_criterion(np_prediction_score,
                                    np_seq_relationship_score,
                                    np_masked_lm_labels,
                                    np_next_sentence_labels, masked_lm_scale,
                                    np_masked_lm_weights)
        loss = self.criterion(prediction_score, seq_relationship_score,
                              masked_lm_labels, next_sentence_labels,
                              masked_lm_scale, masked_lm_weights)

        self.check_output_equal(np_loss, loss.numpy()[0])


class TestBigBirdPretrainingCriterionNotUseNSP(
        TestBigBirdPretrainingCriterionUseNSP):

    def setUp(self):
        self.config['vocab_size'] = 1024
        self.config['use_nsp'] = False
        self.criterion = BigBirdPretrainingCriterion(**self.config)
        self.np_criterion = NpBigBirdPretrainingCriterion(**self.config)


class TestBigBirdFromPretrain(CommonTest):

    @slow
    def test_bigbird_base_uncased(self):
        model = BigBirdModel.from_pretrained('bigbird-base-uncased',
                                             attn_dropout=0.0,
                                             hidden_dropout_prob=0.0)
        self.config = copy.deepcopy(model.config)
        self.config['seq_len'] = 512
        self.config['batch_size'] = 3

        rand_mask_idx_list, input_ids, _ = create_input_data(self.config, 102)
        input_ids = paddle.to_tensor(input_ids)
        rand_mask_idx_list = paddle.to_tensor(rand_mask_idx_list)
        output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)

        expected_seq_shape = (self.config['batch_size'], self.config['seq_len'],
                              self.config['hidden_size'])
        expected_pooled_shape = (self.config['batch_size'],
                                 self.config['hidden_size'])
        self.check_output_equal(output[0].numpy().shape, expected_seq_shape)
        self.check_output_equal(output[1].numpy().shape, expected_pooled_shape)

        expected_seq_slice = np.array([[0.06685783, 0.01576832, -0.14448889],
                                       [0.16531630, 0.00974050, -0.15113291],
                                       [0.08514148, -0.01252885, -0.12458798]])
        # There's output diff about 1e-4 between cpu and gpu
        self.check_output_equal(output[0].numpy()[0, 0:3, 0:3],
                                expected_seq_slice,
                                atol=1e-4)

        expected_pooled_slice = np.array([[0.78695089, 0.87273526, -0.88046724],
                                          [0.66016346, 0.74889791, -0.76608104],
                                          [0.15944470, 0.25242448,
                                           -0.34336662]])
        self.check_output_equal(output[1].numpy()[0:3, 0:3],
                                expected_pooled_slice,
                                atol=1e-4)


if __name__ == "__main__":
    unittest.main()
