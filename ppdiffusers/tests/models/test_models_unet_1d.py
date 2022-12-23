# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
from test_modeling_common import ModelTesterMixin

from ppdiffusers import UNet1DModel
from ppdiffusers.utils import floats_tensor, slow


class UNet1DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet1DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_features = 14
        seq_len = 16

        noise = floats_tensor((batch_size, num_features, seq_len))
        time_step = paddle.to_tensor([10] * batch_size)

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (4, 14, 16)

    @property
    def output_shape(self):
        return (4, 14, 16)

    def test_ema_training(self):
        pass

    def test_training(self):
        pass

    def test_determinism(self):
        super().test_determinism()

    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()

    def test_from_save_pretrained(self):
        super().test_from_save_pretrained()

    def test_model_from_pretrained(self):
        super().test_model_from_pretrained()

    def test_output(self):
        super().test_output()

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64, 128, 256),
            "in_channels": 14,
            "out_channels": 14,
            "time_embedding_type": "positional",
            "use_timestep_embedding": True,
            "flip_sin_to_cos": False,
            "freq_shift": 1.0,
            "out_block_type": "OutConv1DBlock",
            "mid_block_type": "MidResTemporalBlock1D",
            "down_block_types": ("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            "up_block_types": ("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
            "act_fn": "mish",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="unet"
        )
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNet1DModel.from_pretrained("bglick13/hopper-medium-v2-value-function-hor32", subfolder="unet")
        paddle.seed(0)

        num_features = model.in_channels
        seq_len = 16
        noise = paddle.randn((1, seq_len, num_features)).transpose(
            [0, 2, 1]
        )  # match original, we can update values and remove
        time_step = paddle.full((num_features,), 0)

        with paddle.no_grad():
            output = model(noise, time_step).sample.transpose([0, 2, 1])

        output_slice = output[0, -3:, -3:].flatten()

        expected_output_slice = paddle.to_tensor(
            [
                -0.28575825691223145,
                -0.9908179044723511,
                0.2976352870464325,
                -0.8677193522453308,
                -0.2177833616733551,
                0.08095616102218628,
                -0.587175726890564,
                0.32997289299964905,
                -0.1742163598537445,
            ]
        )

        self.assertTrue(paddle.allclose(output_slice, expected_output_slice, rtol=1e-3))

    def test_forward_with_norm_groups(self):
        # Not implemented yet for this UNet
        pass

    @slow
    def test_unet_1d_maestro(self):
        model_id = "harmonai/maestro-150k"
        model = UNet1DModel.from_pretrained(model_id, subfolder="unet")

        sample_size = 65536
        noise = paddle.sin(paddle.arange(sample_size)[None, None, :].cast("float32").tile([1, 2, 1]))
        timestep = paddle.to_tensor([1])

        with paddle.no_grad():
            output = model(noise, timestep).sample

        output_sum = output.abs().sum()
        output_max = output.abs().max()

        assert (output_sum - 15417.94335938).abs() < 4e-2
        assert (output_max - 1.02229333).abs() < 4e-4


class UNetRLModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet1DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_features = 14
        seq_len = 16

        noise = floats_tensor((batch_size, num_features, seq_len))
        time_step = paddle.to_tensor([10] * batch_size)

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (4, 14, 16)

    @property
    def output_shape(self):
        return (4, 14, 1)

    def test_determinism(self):
        super().test_determinism()

    def test_outputs_equivalence(self):
        super().test_outputs_equivalence()

    def test_from_save_pretrained(self):
        super().test_from_save_pretrained()

    def test_model_from_pretrained(self):
        super().test_model_from_pretrained()

    def test_output(self):
        # UNetRL is a value-function is different output shape
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = list((inputs_dict["sample"].shape[0], 1))
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_ema_training(self):
        pass

    def test_training(self):
        pass

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "in_channels": 14,
            "out_channels": 14,
            "down_block_types": ["DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"],
            "up_block_types": [],
            "out_block_type": "ValueFunction",
            "mid_block_type": "ValueFunctionMidBlock1D",
            "block_out_channels": [32, 64, 128, 256],
            "layers_per_block": 1,
            "downsample_each_block": True,
            "use_timestep_embedding": True,
            "freq_shift": 1.0,
            "flip_sin_to_cos": False,
            "time_embedding_type": "positional",
            "act_fn": "mish",
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        value_function, vf_loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="value_function"
        )
        self.assertIsNotNone(value_function)
        self.assertEqual(len(vf_loading_info["missing_keys"]), 0)

        image = value_function(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        value_function, vf_loading_info = UNet1DModel.from_pretrained(
            "bglick13/hopper-medium-v2-value-function-hor32", output_loading_info=True, subfolder="value_function"
        )
        paddle.seed(0)

        num_features = value_function.in_channels
        seq_len = 14
        noise = paddle.randn((1, seq_len, num_features)).transpose(
            [0, 2, 1]
        )  # match original, we can update values and remove
        time_step = paddle.full((num_features,), 0)

        with paddle.no_grad():
            output = value_function(noise, time_step).sample

        expected_output_slice = paddle.to_tensor([291.51135254] * seq_len)
        self.assertTrue(paddle.allclose(output.flatten(), expected_output_slice, rtol=1e-3))

    def test_forward_with_norm_groups(self):
        # Not implemented yet for this UNet
        pass
