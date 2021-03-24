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
from paddlenlp.transformers import BigBirdForSequenceClassification, \
    BigBirdPretrainingCriterion, BigBirdForPretraining, BigBirdModel
from paddlenlp.transformers import create_bigbird_rand_mask_idx_list

from common_test import CommonTest


def create_input_data(config):
    np.random.seed(102)
    rand_mask_idx_list = create_bigbird_rand_mask_idx_list(
        config["num_layers"], config["seq_len"], config["seq_len"],
        config["nhead"], config["block_size"], config["window_size"],
        config["num_global_blocks"], config["num_rand_blocks"], config["seed"])
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
    return rand_mask_idx_list, input_ids, masked_lm_positions


def create_bigbird_model(config, filename, test_model):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(dir_path, '{}.pdparams'.format(filename))
    if not os.path.exists(model_file):
        paddle.seed(102)
        bigbird = BigBirdModel(**config)
        model = test_model(bigbird)
        paddle.save(model.state_dict(), model_file)
    return model_file


class TestBigBirdForSequenceClassification(CommonTest):
    def set_config(self):
        self.config = {
            "num_layers": 2,
            "vocab_size": 1024,
            "nhead": 12,
            "attn_dropout": 0.0,
            "dim_feedforward": 1024,
            "activation": "gelu",
            "normalize_before": False,
            "block_size": 16,
            "window_size": 3,
            "num_global_blocks": 2,
            "num_rand_blocks": 3,
            "seed": None,
            "pad_token_id": 0,
            "hidden_size": 768,
            "hidden_dropout_prob": 0.0,
            "max_position_embeddings": 2048,
            "type_vocab_size": 2,
            "num_labels": 2,
            "initializer_range": 0.02,
            "seq_len": 1024,
            "batch_size": 2
        }
        self.model_file = create_bigbird_model(self.config, "test_bigbird_cls",
                                               BigBirdForSequenceClassification)
        self.rand_mask_idx_list, self.input_ids, _ = create_input_data(
            self.config)

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = BigBirdForSequenceClassification(bigbird)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        expected_output = np.array(
            [[0.38314182, -0.13412490], [0.32075390, 0.07187212]])

        self.check_output_equal(output, expected_output)


class TestBigBirdForPretraining(CommonTest):
    def set_config(self):
        self.config = {
            "num_layers": 2,
            "vocab_size": 1024,
            "nhead": 12,
            "attn_dropout": 0.0,
            "dim_feedforward": 1024,
            "activation": "gelu",
            "normalize_before": False,
            "block_size": 16,
            "window_size": 3,
            "num_global_blocks": 2,
            "num_rand_blocks": 3,
            "seed": None,
            "pad_token_id": 0,
            "hidden_size": 768,
            "hidden_dropout_prob": 0.0,
            "max_position_embeddings": 2048,
            "type_vocab_size": 2,
            "num_labels": 2,
            "initializer_range": 0.02,
            "seq_len": 1024,
            "batch_size": 2
        }
        self.model_file = create_bigbird_model(
            self.config, "test_bigbird_pretrain", BigBirdForPretraining)
        self.rand_mask_idx_list, self.input_ids, self.masked_lm_positions = create_input_data(
            self.config)

    def test_forward(self):
        bigbird = BigBirdModel(**self.config)
        model = BigBirdForPretraining(bigbird)
        state_dict = paddle.load(self.model_file)
        model.set_state_dict(state_dict)
        input_ids = paddle.to_tensor(self.input_ids)
        rand_mask_idx_list = paddle.to_tensor(self.rand_mask_idx_list)
        masked_positions = paddle.to_tensor(self.masked_lm_positions)
        output = model(
            input_ids,
            rand_mask_idx_list=rand_mask_idx_list,
            masked_positions=masked_positions)

        expected_prediction_scores_sum = np.array([-0.00156948])
        expected_prediction_scores_abs_sum = np.array([0.44451436])
        expected_seq_relationship_score = np.array(
            [[-0.23682573, -0.78529185], [0.20035000, -0.75405741]])

        prediction_scores_sum = output[0].sum().numpy() / output[0].size
        prediction_scores_abs_sum = output[0].abs().sum().numpy() / output[
            0].size
        seq_relationship_score = output[1].numpy()

        self.check_output_equal(prediction_scores_sum,
                                expected_prediction_scores_sum)
        self.check_output_equal(prediction_scores_abs_sum,
                                expected_prediction_scores_abs_sum)
        self.check_output_equal(seq_relationship_score,
                                expected_seq_relationship_score)


if __name__ == "__main__":
    unittest.main()
