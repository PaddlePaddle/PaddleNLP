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

import unittest

import numpy as np
import paddle

from .modeling import GLM130BConfig, GLM130BModel


class GLM130BModelTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = self.get_config()
        self.model = GLM130BModel(self.config)

    def get_config(self):
        return GLM130BConfig(
            **{
                "hidden_size": 192,
                "inner_hidden_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "length_per_sample": 100,
                "max_length": 128,
                "vocab_size_base": 768,
                "activation": "geglu",
                "layernorm_epsilon": 1e-5,
                "paddle_dtype": "float16",
                "attention_dropout_prob": 0,
                "attention_scale": True,
                "embedding_dropout_prob": 0,
                "initializer_range": 0.0052,
                "output_dropout_prob": 0,
                "output_predict": True,
                "position_encoding_2d": False,
                "recompute": False,
                "vocab_size": 150528,
            }
        )

    def create_inputs(self):
        input_ids = paddle.to_tensor(np.load("torch_cache/torch_input_ids.npy"))
        position_ids = paddle.to_tensor(np.load("torch_cache/torch_position_ids.npy"))
        attention_mask = paddle.to_tensor(np.load("torch_cache/torch_attention_mask.npy"))
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def create_model_state(self):
        import torch

        model = torch.load("torch_cache/model.pt")
        keys = list(model.keys())
        new_state = {}
        for k in keys:
            new_state[k] = paddle.to_tensor(model[k].detach().cpu().numpy())
            if "weight" in k and "word_embedding" not in k and "layernorm" not in k:
                new_state[k] = new_state[k].transpose([1, 0])
        return new_state

    def test_model(self):
        paddle.set_default_dtype("float16")
        config = self.get_config()
        inputs = self.create_inputs()

        model = GLM130BModel(config)
        model_state = self.create_model_state()
        model.set_state_dict(model_state)
        model.eval()

        results = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        model.save_pretrained("local_random_glm")
        self.assertAlmostEqual(results.logits.abs().mean().item(), 0.0003113746643066406, places=7)


if __name__ == "__main__":
    unittest.main()
