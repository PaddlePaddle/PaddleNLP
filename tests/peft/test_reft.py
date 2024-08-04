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

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.peft.reft.pareft import (
    LoreftIntervention,
    LowRankRotateLayer,
    ReftConfig,
    ReftDataCollator,
    ReftTrainer,
    TinyIntervention,
    get_reft_model,
)
from paddlenlp.peft.reft.pareft.dataset import (
    LoReftSupervisedDataset,
    ReftDataset,
    get_intervention_locations,
    parse_positions,
)
from paddlenlp.peft.reft.pareft.predict import do_predict
from paddlenlp.peft.reft.pareft.reft_model import ReftModel
from paddlenlp.peft.reft.pavenv.models.basic_utils import get_type_from_string
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


class TestReftDataCollator(unittest.TestCase):
    def test_call(self):
        model_name = "__internal_testing__/tiny-random-llama"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=512,
            padding_side="right",
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
        reft_data_collator = ReftDataCollator(data_collator)
        instances = [
            {
                "input_ids": paddle.to_tensor([[1, 2, 3], [4, 5, 6]]),
                "intervention_locations": paddle.to_tensor([[0, 1, 0], [1, 0, 1]]),
            },
            {
                "input_ids": paddle.to_tensor([[7, 8, 9], [10, 11, 12]]),
                "intervention_locations": paddle.to_tensor([[1, 0, 1], [0, 1, 0]]),
            },
        ]

        batch_inputs = reft_data_collator(instances)

        self.assertIn("input_ids", batch_inputs)
        self.assertIn("intervention_locations", batch_inputs)
        self.assertIsInstance(batch_inputs["input_ids"], paddle.Tensor)
        self.assertIsInstance(batch_inputs["intervention_locations"], paddle.Tensor)


class TestBasicUtils(unittest.TestCase):
    def test_get_type_from_string(self):
        class_str = "pareft.interventions.LoreftIntervention"
        cls = get_type_from_string(class_str)
        self.assertIsInstance(cls, type(LoreftIntervention))

    def test_parse_positions(self):
        positions = "f7+l7"
        self.assertEqual(parse_positions(positions), (7, 7))
        positions = "f7"
        self.assertEqual(parse_positions(positions), (7, 0))
        positions = "l7"
        self.assertEqual(parse_positions(positions), (0, 7))

    def test_get_intervention_locations(self):
        kwargs = {"last_position": 10, "positions": "f7+l7", "num_interventions": 1}
        intervention_locations1 = get_intervention_locations(**kwargs)
        print(intervention_locations1)
        kwargs = {"last_position": 10, "first_n": 7, "last_n": 7, "num_interventions": 1}
        intervention_locations2 = get_intervention_locations(**kwargs)
        self.assertEqual(intervention_locations1, intervention_locations2)

    def test_reft_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "__internal_testing__/tiny-random-llama",
            model_max_length=512,
            padding_side="right",
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id
        train_ds = LoReftSupervisedDataset(
            "./tests/fixtures/llm/data",
            tokenizer,
            data_split="train",
            seed=42,
            **{
                "num_interventions": 2,
                "position": "f7+l7",
                "trigger_tokens": "LLM Response: ",
            },
        )
        self.assertIsInstance(train_ds, ReftDataset)


class TestLoReftIntervention(unittest.TestCase):
    def setUp(self):
        self.kwargs = {
            "embed_dim": 64,
            "low_rank_dimension": 4,
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

    def test_load_state_dict(self):
        model = LoreftIntervention(**self.kwargs)
        state_dict = {
            "learned_source.weight": paddle.randn([64, 4]),
            "learned_source.bias": paddle.zeros([4]),
            "rotate_layer.weight": paddle.randn([64, 4]),
        }
        model.load_state_dict(state_dict)
        self.assertTrue(paddle.allclose(model.learned_source.weight.data, state_dict["learned_source.weight"]))
        self.assertTrue(paddle.allclose(model.learned_source.bias.data, state_dict["learned_source.bias"]))
        self.assertTrue(
            paddle.allclose(
                model.rotate_layer.weight[:, : state_dict["rotate_layer.weight"].shape[-1]],
                state_dict["rotate_layer.weight"],
            )
        )


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


class TestReftModelTrain(unittest.TestCase):
    def test_reft_model_train(self):
        model_name = "__internal_testing__/tiny-random-llama"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=128,
            padding_side="right",
        )
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype="bfloat16")
        intervention_dtype = "bfloat16"
        layers = [int(l) for l in range(1)]
        representations = [
            {
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": LoreftIntervention(
                    embed_dim=768,
                    low_rank_dimension=4,
                    dropout=0.00,
                    dtype=intervention_dtype,
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
        reft_model.model.train()
        train_ds = LoReftSupervisedDataset(
            "./tests/fixtures/llm/data",
            tokenizer,
            data_split="train",
            seed=42,
            **{
                "num_interventions": len(layers),
                "position": "f5+l5",
                "trigger_tokens": "LLM Response: ",
            },
        )
        dev_ds = LoReftSupervisedDataset(
            "./tests/fixtures/llm/data",
            tokenizer,
            data_split="dev",
            seed=42,
            **{
                "num_interventions": len(layers),
                "position": "f5+l5",
                "trigger_tokens": "LLM Response: ",
            },
        )
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding="longest"
        )
        data_collator = ReftDataCollator(data_collator=data_collator_fn)
        trainer = ReftTrainer(
            model=reft_model,
            tokenizer=tokenizer,
            # args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
            eval_dataset=None,
            compute_metrics=None,
        )
        trainer.train()
        do_predict(
            intervenable=reft_model,
            tokenizer=tokenizer,
            eval_dataset=dev_ds,
            data_items=dev_ds.raw_dataset,
            batch_size=1,
        )


if __name__ == "__main__":
    unittest.main()
