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
import unittest
from tempfile import TemporaryDirectory

import paddle

from paddlenlp.layers import LoRALinearLayer


class TestLoraLayer(unittest.TestCase):
    def test_forward(self):
        lora_layer = LoRALinearLayer(in_features=16, out_features=8, r=4)
        x = paddle.randn([2, 16], "float32")
        output = lora_layer(x)
        self.assertFalse(lora_layer.lora_A.stop_gradient)
        self.assertFalse(lora_layer.lora_B.stop_gradient)
        self.assertTrue(lora_layer.weight.stop_gradient)
        self.assertTrue(lora_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 16], "float32")
        lora_layer = LoRALinearLayer(in_features=16, out_features=8, r=4)
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
            lora_layer = LoRALinearLayer(in_features=16, out_features=8, r=4)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(lora_layer.state_dict(), weights_path)
            new_lora_layer = LoRALinearLayer(in_features=16, out_features=8, r=4)
            state_dict = paddle.load(weights_path)
            new_lora_layer.set_dict(state_dict)
            x = paddle.randn([2, 16], "float32")
            self.assertTrue(paddle.allclose(new_lora_layer(x), lora_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = LoRALinearLayer(in_features=16, out_features=8)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            lora_layer_r0 = LoRALinearLayer(in_features=16, out_features=8, r=0)
            lora_layer_r4 = LoRALinearLayer(in_features=16, out_features=8, r=4)
            lora_layer_r0.set_dict(state_dict)
            lora_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 16], "float32")
            self.assertTrue(paddle.allclose(lora_layer_r0(x), regular_linear(x)))
            self.assertFalse(paddle.allclose(lora_layer_r4(x), regular_linear(x)))
