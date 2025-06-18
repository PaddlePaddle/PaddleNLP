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
from parameterized import parameterized

from paddlenlp.peft.lora import LoRAConfig, LoRALinear, LoRAModel
from paddlenlp.transformers import AutoModel, BertModel


class TestNolaLayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            LoRALinear(
                in_features=16, out_features=8, r=0, lora_dropout=0.1, lora_alpha=8, nola=True, nola_basis_num=2
            )

    def test_forward(self):
        nola_layer = LoRALinear(
            in_features=16, out_features=8, r=4, lora_dropout=0.1, lora_alpha=8, nola=True, nola_basis_num=2
        )
        x = paddle.randn([2, 4, 16], "float32")
        output = nola_layer(x)
        self.assertFalse(nola_layer.lora_A.stop_gradient)
        self.assertTrue(nola_layer.weight.stop_gradient)
        self.assertFalse(nola_layer.bias.stop_gradient)
        self.assertEqual(output.shape, [2, 4, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 4, 16], "float32")
        nola_layer = LoRALinear(in_features=16, out_features=8, r=4, nola=True, nola_basis_num=2)
        nola_layer.train()
        train_result = nola_layer(x)
        train_weight = copy.deepcopy(nola_layer.weight)  # deep copy since this is a pointer
        nola_layer.eval()
        eval_result = nola_layer(x)
        eval_weight = nola_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertTrue(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            nola_layer = LoRALinear(in_features=16, out_features=8, r=4, nola=True, nola_basis_num=2)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(nola_layer.state_dict(), weights_path)
            new_nola_layer = LoRALinear(in_features=16, out_features=8, r=4, nola=True, nola_basis_num=2)
            state_dict = paddle.load(weights_path)
            new_nola_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_nola_layer(x), nola_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=8)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear
            nola_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, nola=True, nola_basis_num=2)
            nola_layer_r4 = LoRALinear(in_features=16, out_features=8, r=4, nola=True, nola_basis_num=2)
            nola_layer_r8.set_dict(state_dict)
            nola_layer_r4.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(nola_layer_r8(x), regular_linear(x)))
            self.assertTrue(paddle.allclose(nola_layer_r4(x), regular_linear(x)))

    def test_merge(self):
        nola_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, nola=True, nola_basis_num=2)
        nola_layer_r8.merge()

    def test_unmerge(self):
        nola_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8, nola=True, nola_basis_num=2)
        nola_layer_r8.merged = True
        nola_layer_r8.unmerge()
        nola_layer_r8 = LoRALinear(in_features=16, out_features=8, r=8)
        nola_layer_r8.merged = True
        nola_layer_r8.unmerge()


class TestNolaModel(unittest.TestCase):
    def test_nola_model_restore(self):
        nola_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            head_dim=2,
            nola=True,
            nola_basis_num=2,
        )
        model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.eval()
        original_results_1 = model(input_ids)
        nola_model = LoRAModel(model, nola_config)
        restored_model = nola_model.restore_original_model()
        restored_model.eval()
        original_results_2 = restored_model(input_ids)
        self.assertIsNotNone(original_results_1)
        self.assertIsNotNone(original_results_2)
        self.assertIsInstance(restored_model, BertModel)
        self.assertTrue(paddle.allclose(original_results_1[0], original_results_2[0]))

    @parameterized.expand([(None,), ("all",), ("lora",)])
    def test_nola_model_constructor(self, bias):
        nola_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
            enable_lora_list=[None, [True, False]],
            trainable_bias=bias,
            head_dim=2,
            nola=True,
            nola_basis_num=2,
        )
        # turn off plm dropout for to test train vs test
        model = AutoModel.from_pretrained(
            "__internal_testing__/tiny-random-bert", hidden_dropout_prob=0, attention_probs_dropout_prob=0
        )
        nola_model = LoRAModel(model, nola_config)
        nola_model.mark_only_lora_as_trainable()
        for name, weight in nola_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in nola_config.target_modules]):
                if "lora" in name:
                    self.assertFalse(weight.stop_gradient)
                elif "bias" in name and bias in ["lora", "all"]:
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)
            else:
                if "bias" in name and bias == "all":
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        nola_model.train()
        train_forward_results = nola_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        nola_model.eval()
        eval_forward_results = nola_model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        self.assertTrue(paddle.allclose(train_forward_results[0], eval_forward_results[0]))

    def test_nola_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
            nola_config = LoRAConfig(
                target_modules=[".*q_proj.*", ".*v_proj.*"], r=4, lora_alpha=8, nola=True, nola_basis_num=2
            )
            model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
            nola_model = LoRAModel(model, nola_config)
            nola_model.eval()
            original_results = nola_model(input_ids)
            nola_model.save_pretrained(tempdir)

            loaded_nola_model = LoRAModel.from_pretrained(model, tempdir)
            loaded_nola_model.eval()
            loaded_results = loaded_nola_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_nola_model = LoRAModel.from_pretrained(model, tempdir, lora_config=nola_config)
            config_loaded_nola_model.eval()
            config_loaded_results = config_loaded_nola_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))

    def test_lora_module_raise_exception(self):
        nola_config = LoRAConfig(
            target_modules=[".*norm1.*"], r=4, lora_alpha=8, enable_lora_list=None, nola=True, nola_basis_num=2
        )
        model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
        with self.assertRaises(ValueError):
            LoRAModel(model, nola_config)


class TestNolaConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            nola_config = LoRAConfig()
            nola_config.save_pretrained(tempdir)
            loaded_nola_config = LoRAConfig.from_pretrained(tempdir)
            self.assertEqual(nola_config, loaded_nola_config)
