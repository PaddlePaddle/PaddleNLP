# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle

from paddlenlp.layers import (
    LoRAConfig,
    LoRALinear,
    get_lora_model,
    mark_only_lora_as_trainable,
)
from paddlenlp.transformers import AutoModel


class TestLoraLayer(unittest.TestCase):
    def test_forward(self):
        lora_layer = LoRALinear(in_features=16, out_features=8, r=4, lora_dropout=0.1, lora_alpha=8)
        x = paddle.randn([2, 16], "float32")
        output = lora_layer(x)
        self.assertFalse(lora_layer.lora_A.stop_gradient)
        self.assertFalse(lora_layer.lora_B.stop_gradient)
        self.assertTrue(lora_layer.weight.stop_gradient)
        self.assertTrue(lora_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 16], "float32")
        lora_layer = LoRALinear(in_features=16, out_features=8, r=4)
        lora_layer.train()
        train_result = lora_layer(x)
        train_weight = copy.deepcopy(lora_layer.weight)  # deep copy since this is a pointer
        lora_layer.eval()
        eval_result = lora_layer(x)
        eval_weight = lora_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertFalse(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            lora_layer = LoRALinear(in_features=16, out_features=8, r=4)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(lora_layer.state_dict(), weights_path)
            new_lora_layer = LoRALinear(in_features=16, out_features=8, r=4)
            state_dict = paddle.load(weights_path)
            new_lora_layer.set_dict(state_dict)
            x = paddle.randn([2, 16], "float32")
            self.assertTrue(paddle.allclose(new_lora_layer(x), lora_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=8)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            lora_layer_r0 = LoRALinear(in_features=16, out_features=8, r=0)
            lora_layer_r4 = LoRALinear(in_features=16, out_features=8, r=4)
            lora_layer_r0.set_dict(state_dict)
            lora_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 16], "float32")
            self.assertTrue(paddle.allclose(lora_layer_r0(x), regular_linear(x)))
            self.assertFalse(paddle.allclose(lora_layer_r4(x), regular_linear(x)))


class TestLoraModel(unittest.TestCase):
    def test_get_lora_model(self):
        lora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            merge_weights=True,
        )
        # turn off plm dropout for to test train vs test
        model = AutoModel.from_pretrained(
            "__internal_testing__/tiny-random-bert", hidden_dropout_prob=0, attention_probs_dropout_prob=0
        )
        lora_model = get_lora_model(model, lora_config)
        mark_only_lora_as_trainable(lora_model)
        state_dict = lora_model.state_dict()
        for weight_name in state_dict:
            is_target_module = False
            for target_module in lora_config.target_modules:
                if re.fullmatch(target_module, weight_name):
                    is_target_module = True
            # if this is a target module, lora weights are trainable, non-lora weights are not
            if is_target_module:
                if "lora" in weight_name:
                    self.assertFalse(state_dict[weight_name].stop_gradient)
                else:
                    self.assertTrue(state_dict[weight_name].stop_gradient)
            # if this is not a target module, all weights are not trainable
            else:
                self.assertTrue(state_dict[weight_name].stop_gradient)
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.train()
        train_forward_results = model(input_ids)
        self.assertIsNotNone(train_forward_results)
        model.eval()
        eval_forward_results = model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        for i, j in zip(train_forward_results, eval_forward_results):
            self.assertTrue(paddle.allclose(i, j))


class TestLoRAConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            lora_config = LoRAConfig()
            lora_config.save_pretrained(tempdir)
            loaded_lora_config = LoRAConfig.from_pretrained(tempdir)
            self.assertEqual(lora_config, loaded_lora_config)
