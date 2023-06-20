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

import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import paddle
from paddle.quantization import QAT, QuantConfig
from paddle.quantization.config import SingleLayerConfig
from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver

from paddlenlp.peft.lora import LoRAConfig, LoRALinear, LoRAModel, QuantedLoRALinear
from paddlenlp.transformers import AutoModel


class TestQuantedLoraLayer(unittest.TestCase):
    def test_forward(self):
        q_config = SingleLayerConfig(weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9), activation=None)
        lora_layer = LoRALinear(in_features=16, out_features=8, r=4, lora_dropout=0.1, lora_alpha=8)
        quant_lora_layer = QuantedLoRALinear(layer=lora_layer, q_config=q_config)
        x = paddle.randn([2, 4, 16], "float32")
        quant_output = quant_lora_layer(x)
        self.assertFalse(quant_lora_layer.lora_A.stop_gradient)
        self.assertFalse(quant_lora_layer.lora_B.stop_gradient)
        self.assertTrue(quant_lora_layer.weight.stop_gradient)
        self.assertFalse(quant_lora_layer.bias.stop_gradient)
        self.assertEqual(quant_output.shape, [2, 4, 8])

    def test_save_load(self):
        with TemporaryDirectory() as tempdir:
            q_config = SingleLayerConfig(weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9), activation=None)
            quant_lora_layer = QuantedLoRALinear(
                layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8), q_config=q_config
            )
            weights_path = os.path.join(tempdir, "model.pdparams")
            paddle.save(quant_lora_layer.state_dict(), weights_path)
            new_quant_lora_layer = QuantedLoRALinear(
                layer=LoRALinear(in_features=16, out_features=8, r=4, lora_alpha=8), q_config=q_config
            )
            state_dict = paddle.load(weights_path)
            new_quant_lora_layer.set_dict(state_dict)
            x = paddle.randn([2, 4, 16], "float32")
            self.assertTrue(paddle.allclose(new_quant_lora_layer(x), quant_lora_layer(x)))


class TestQuantedLoRAModel(unittest.TestCase):
    def test_quant_model_forward(self):
        lora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
            r=4,
            lora_alpha=8,
        )
        model = AutoModel.from_pretrained("__internal_testing__/tiny-random-bert")
        lora_model = LoRAModel(model, lora_config)
        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_type_config(LoRALinear, weight=FakeQuanterWithAbsMaxObserver(moving_rate=0.9))
        qat = QAT(q_config)
        quant_lora_model = qat.quantize(lora_model)
        input_ids = paddle.to_tensor(np.random.randint(100, 200, [1, 20]))
        original_model_outputs = lora_model(input_ids)[0]
        quant_model_outputs = quant_lora_model(input_ids)[0]
        self.assertEqual(original_model_outputs.shape, quant_model_outputs.shape)
