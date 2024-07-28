# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddlenlp.peft.reft.pareft import (
    LoreftIntervention,
    LowRankRotateLayer,
    ReftConfig,
    TinyIntervention,
    get_reft_model,
)
from paddlenlp.peft.reft.pareft.reft_model import ReftModel
from paddlenlp.peft.reft.pavenv.models.basic_utils import get_type_from_string
from paddlenlp.transformers import AutoModelForCausalLM


class TestBasicUtils(unittest.TestCase):
    def test_get_type_from_string(self):
        class_str = "pareft.interventions.LoreftIntervention"
        cls = get_type_from_string(class_str)
        self.assertIsInstance(cls, type(LoreftIntervention))


class TestLoReftIntervention(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embed_dim": 256,
            "low_rank_dimension": 64,
            "dtype": paddle.float32,
            "dropout": 0.1,
            "act_fn": "linear",
        }

    def test_initialization(self):
        intervention = LoreftIntervention(**self.kwargs)
        self.assertIsInstance(intervention.rotate_layer, LowRankRotateLayer)
        self.assertIsInstance(intervention.learned_source, paddle.nn.Linear)
        self.assertEqual(intervention.dropout.p, self.kwargs["dropout"])

    def test_forward(self):
        base = paddle.randn([10, self.kwargs["embed_dim"]])
        intervention = LoreftIntervention(**self.kwargs)
        output = intervention.forward(base)
        self.assertEqual(output.shape, base.shape)
        self.assertEqual(output.dtype, self.kwargs["dtype"])


class TestTinyIntervention(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embed_dim": 256,
            "low_rank_dimension": 64,
            "dtype": paddle.float32,
            "dropout": 0.1,
            "act_fn": "relu",
        }

    def test_initialization(self):
        intervention = TinyIntervention(**self.kwargs)
        self.assertEqual(intervention.rank, self.kwargs["low_rank_dimension"])
        self.assertEqual(intervention.hidden_size, self.kwargs["embed_dim"])
        self.assertEqual(intervention.param_A.shape, [self.kwargs["embed_dim"], self.kwargs["low_rank_dimension"]])
        self.assertEqual(intervention.param_B.shape, [self.kwargs["low_rank_dimension"], self.kwargs["embed_dim"]])
        self.assertEqual(intervention.param_a.shape, [self.kwargs["low_rank_dimension"]])
        self.assertEqual(intervention.param_b.shape, [self.kwargs["embed_dim"]])

    def test_forward(self):
        base = paddle.randn([10, self.kwargs["embed_dim"]])
        intervention = TinyIntervention(**self.kwargs)
        output = intervention.forward(base)

        self.assertEqual(output.shape, base.shape)
        self.assertEqual(output.dtype, self.kwargs["dtype"])


class TestReftModel(unittest.TestCase):
    def test_get_reft_model(self):
        model = AutoModelForCausalLM.from_pretrained("__internal_testing__/tiny-random-llama")
        layers = [0]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype="float32",
                    act_fn="linear",
                    device="gpu",
                    add_bias=False,
                ),
            }
            for l in layers
        ]
        reft_config = ReftConfig(representations=representations)
        reft_model = get_reft_model(model, reft_config, set_device=False)
        reft_model.print_trainable_parameters()
        self.assertTrue(type(reft_model), ReftModel)

    def test_reft_model_forward(self):
        model = AutoModelForCausalLM.from_pretrained("__internal_testing__/tiny-random-llama")

        layers = [0]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype="float32",
                    act_fn="linear",
                    device="gpu",
                    add_bias=False,
                ),
            }
            for l in layers
        ]
        reft_config = ReftConfig(representations=representations)
        reft_model = get_reft_model(model, reft_config, set_device=False)
        reft_model.print_trainable_parameters()
        outputs = reft_model.model(**{"input_ids": paddle.randint(low=1, high=100, shape=(5, 10))})
        self.assertTrue(outputs[0].shape, [5, 10, 32000])


if __name__ == "__main__":
    unittest.main()
