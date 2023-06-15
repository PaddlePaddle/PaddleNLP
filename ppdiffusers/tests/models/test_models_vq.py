# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from ppdiffusers_test.test_modeling_common import ModelTesterMixin

from ppdiffusers import VQModel
from ppdiffusers.utils import floats_tensor


class VQModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = VQModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3
        image = floats_tensor((batch_size, num_channels) + sizes)
        return {"sample": image}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 3,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = VQModel.from_pretrained("fusing/vqgan-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)
        image = model(**self.dummy_input)
        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = VQModel.from_pretrained("fusing/vqgan-dummy")
        model.eval()
        paddle.seed(0)
        image = paddle.randn(shape=[1, model.config.in_channels, model.config.sample_size, model.config.sample_size])
        with paddle.no_grad():
            output = model(image).sample
        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(
            [
                -0.027147896587848663,
                -0.41129639744758606,
                -0.17730756103992462,
                -0.5245445370674133,
                -0.2423611730337143,
                -0.3957087993621826,
                -0.16461530327796936,
                -0.06902074813842773,
                -0.01736617460846901,
            ]
        )
        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, atol=0.01))
