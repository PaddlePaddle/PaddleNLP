# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.peft.dislora import DisLoRAConfig, DisLoRALinear, DisLoRAModel
from paddlenlp.transformers import AutoModel, BertModel


class TestDisLoRALayer(unittest.TestCase):
    def test_r_raise_exception(self):
        with self.assertRaises(ValueError):
            DisLoRALinear(in_features=16, out_features=8, r=0, dislora_alpha=8)

    def test_forward(self):
        # r=8, dislora_alpha=12 (1.5 * 8)
        dislora_layer = DisLoRALinear(in_features=16, out_features=8, r=8, dislora_dropout=0.1, dislora_alpha=12)
        x = paddle.randn([2, 4, 16], "float32")
        output = dislora_layer(x)

        # Check the trainable DisLoRA parameters (related to W_res)
        self.assertFalse(dislora_layer.Direc_Ur.weight.stop_gradient)
        self.assertFalse(dislora_layer.Direc_Vhr.weight.stop_gradient)
        self.assertFalse(dislora_layer.Direc_Sr.stop_gradient)
        self.assertFalse(dislora_layer.Direc_Stsd.stop_gradient)

        # Check the frozen TSD parameters
        self.assertTrue(dislora_layer.Direc_Utsd.weight.stop_gradient)
        self.assertTrue(dislora_layer.Direc_Vhtsd.weight.stop_gradient)

        # Check the frozen main branch weights W_prin
        self.assertTrue(dislora_layer.weight.stop_gradient)

        # Check the bias parameters (by default, they should be trainable, but this depends on the configuration)
        if dislora_layer.bias is not None:
            self.assertFalse(dislora_layer.bias.stop_gradient)

        self.assertEqual(output.shape, [2, 4, 8])

    def test_train_eval(self):
        x = paddle.randn([2, 4, 16], "float32")

        dislora_layer = DisLoRALinear(in_features=16, out_features=8, r=8, dislora_alpha=12)
        dislora_layer.train()
        train_result = dislora_layer(x)
        train_weight = copy.deepcopy(dislora_layer.weight)
        dislora_layer.eval()
        eval_result = dislora_layer(x)
        eval_weight = dislora_layer.weight
        self.assertTrue(paddle.allclose(train_result, eval_result))
        self.assertTrue(paddle.allclose(train_weight, eval_weight))

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:

            dislora_layer = DisLoRALinear(in_features=16, out_features=8, r=8, dislora_alpha=12)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(dislora_layer.state_dict(), weights_path)

            new_dislora_layer = DisLoRALinear(in_features=16, out_features=8, r=8, dislora_alpha=12)
            state_dict = paddle.load(weights_path)
            new_dislora_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_dislora_layer(x), dislora_layer(x)))

    def test_load_regular_linear(self):
        with TemporaryDirectory() as tempdir:
            regular_linear = paddle.nn.Linear(in_features=16, out_features=12)
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(regular_linear.state_dict(), weights_path)
            state_dict = paddle.load(weights_path)
            # should be identical to regular linear

            dislora_layer_r8 = DisLoRALinear(
                in_features=16, out_features=12, r=8, dislora_alpha=12, init_lora_weights=False
            )

            dislora_layer_r10 = DisLoRALinear(
                in_features=16, out_features=12, r=10, dislora_alpha=15, init_lora_weights=False
            )

            # Load regular linear weights first
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in ["weight", "bias"]}
            dislora_layer_r8.set_dict(filtered_state_dict)
            dislora_layer_r10.set_dict(filtered_state_dict)

            # Then perform SVD initialization
            dislora_layer_r8._init_lora_weights()
            dislora_layer_r10._init_lora_weights()

            x = paddle.randn([2, 4, 16], "float32")

            diff_r8 = paddle.abs(dislora_layer_r8(x) - regular_linear(x))
            print(f"R8 - Max diff: {paddle.max(diff_r8).item():.6e}, Mean diff: {paddle.mean(diff_r8).item():.6e}")
            self.assertTrue(paddle.allclose(dislora_layer_r8(x), regular_linear(x), atol=2e-3))
            # Update variable name
            self.assertTrue(paddle.allclose(dislora_layer_r10(x), regular_linear(x), atol=2e-3))


class TestDisLoRAModel(unittest.TestCase):
    def test_dislora_model_restore(self):

        dislora_config = DisLoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=8,
            dislora_alpha=12,
            base_model_name_or_path="__internal_testing__/tiny-random-bert",
        )
        model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        model.eval()
        original_results_1 = model(input_ids)
        dislora_model = DisLoRAModel(model, dislora_config)
        restored_model = dislora_model.restore_original_model()
        restored_model.eval()
        original_results_2 = restored_model(input_ids)
        self.assertIsNotNone(original_results_1)
        self.assertIsNotNone(original_results_2)
        self.assertIsInstance(restored_model, BertModel)
        self.assertTrue(paddle.allclose(original_results_1[0], original_results_2[0]))

    @parameterized.expand([(None,), ("all",), ("dislora",)])
    def test_dislora_model_constructor(self, bias):

        dislora_config = DisLoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=8,
            dislora_alpha=12,
            trainable_bias=bias,
            base_model_name_or_path="__internal_testing__/tiny-random-bert",
        )
        model = AutoModel.from_pretrained(
            "__internal_testing__/tiny-random-bert", hidden_dropout_prob=0, attention_probs_dropout_prob=0
        )
        dislora_model = DisLoRAModel(model, dislora_config)
        dislora_model.mark_only_dislora_as_trainable()
        for name, weight in dislora_model.state_dict().items():
            if any([re.fullmatch(target_module, name) for target_module in dislora_config.target_modules]):
                if any(
                    [dislora_param in name for dislora_param in ["Direc_Ur", "Direc_Sr", "Direc_Vhr", "Direc_Stsd"]]
                ):
                    self.assertFalse(weight.stop_gradient)
                elif any([tsd_param in name for tsd_param in ["Direc_Utsd", "Direc_Vhtsd"]]):
                    self.assertTrue(weight.stop_gradient)
                elif "bias" in name and bias in ["dislora", "all"]:
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)
            else:
                if "bias" in name and bias == "all":
                    self.assertFalse(weight.stop_gradient)
                else:
                    self.assertTrue(weight.stop_gradient)

        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        dislora_model.train()
        train_forward_results = dislora_model(input_ids)
        self.assertIsNotNone(train_forward_results)
        dislora_model.eval()
        eval_forward_results = dislora_model(input_ids)
        self.assertIsNotNone(eval_forward_results)
        self.assertTrue(paddle.allclose(train_forward_results[0], eval_forward_results[0]))

    def test_dislora_model_save_load(self):
        with TemporaryDirectory() as tempdir:
            input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))

            dislora_config = DisLoRAConfig(
                target_modules=[".*q_proj.*", ".*v_proj.*"],
                r=8,
                dislora_alpha=12,
                base_model_name_or_path="__internal_testing__/tiny-random-bert",
            )
            model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
            dislora_model = DisLoRAModel(model, dislora_config)
            dislora_model.eval()
            original_results = dislora_model(input_ids)
            dislora_model.save_pretrained(tempdir)

            loaded_dislora_model = DisLoRAModel.from_pretrained(model, tempdir)
            loaded_dislora_model.eval()
            loaded_results = loaded_dislora_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], loaded_results[0]))

            config_loaded_dislora_model = DisLoRAModel.from_pretrained(model, tempdir, dislora_config=dislora_config)
            config_loaded_dislora_model.eval()
            config_loaded_results = config_loaded_dislora_model(input_ids)
            self.assertTrue(paddle.allclose(original_results[0], config_loaded_results[0]))

    def test_dislora_module_raise_exception(self):

        dislora_config = DisLoRAConfig(
            target_modules=[".*norm1.*"],
            r=8,
            dislora_alpha=12,
            base_model_name_or_path="__internal_testing__/tiny-random-bert",
        )
        model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
        with self.assertRaises(ValueError):
            DisLoRAModel(model, dislora_config)


class TestDisLoRAConfig(unittest.TestCase):
    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            # Set r and dislora_alpha explicitly
            dislora_config = DisLoRAConfig(target_modules=["test"], r=8, dislora_alpha=12)
            dislora_config.save_pretrained(tempdir)
            loaded_dislora_config = DisLoRAConfig.from_pretrained(tempdir)
            self.assertEqual(dislora_config.r, loaded_dislora_config.r)
            self.assertEqual(dislora_config.dislora_alpha, loaded_dislora_config.dislora_alpha)
            self.assertEqual(dislora_config.dash_flag, loaded_dislora_config.dash_flag)
            self.assertEqual(dislora_config.s_tsd, loaded_dislora_config.s_tsd)
