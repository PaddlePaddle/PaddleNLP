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

import gc
import math
import unittest

import paddle
from parameterized import parameterized
from test_modeling_common import ModelTesterMixin

from ppdiffusers import UNet2DConditionModel, UNet2DModel
from ppdiffusers.utils import (
    floats_tensor,
    load_ppnlp_numpy,
    logging,
    paddle_all_close,
    slow,
)

logger = logging.get_logger(__name__)


class Unet2DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "attention_head_dim": None,
            "out_channels": 3,
            "in_channels": 3,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict


class UNetLDMModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": (32, 64),
            "attention_head_dim": 32,
            "down_block_types": ("DownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "UpBlock2D"),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)

        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()

        noise = paddle.randn(
            [1, model.config.in_channels, model.config.sample_size, model.config.sample_size],
            generator=paddle.Generator().manual_seed(0),
        )
        time_step = paddle.to_tensor([10] * noise.shape[0])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -1, -3:, -3:].flatten()

        expected_output_slice = paddle.to_tensor(
            [
                0.43856096267700195,
                -10.29347038269043,
                -9.609537124633789,
                -8.39902114868164,
                -16.292064666748047,
                -13.075122833251953,
                -9.303834915161133,
                -13.698592185974121,
                -10.529990196228027,
            ]
        )

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-3))


class UNet2DConditionModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DConditionModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])
        encoder_hidden_states = floats_tensor((batch_size, 4, 32))

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 32, 32)

    @property
    def output_shape(self):
        return (4, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    # TODO
    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        assert not model.is_gradient_checkpointing and model.training
        paddle.seed(0)
        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.clear_gradients()

        labels = paddle.randn(out.shape, dtype=out.dtype)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_dict(model.state_dict())
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.clear_gradients()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        with paddle.no_grad():
            for name, param in named_params.items():
                self.assertTrue(paddle_all_close(param.grad, named_params_2[name].grad, atol=5e-5))

    def test_model_with_attention_head_dim_tuple(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_with_use_linear_projection(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["use_linear_projection"] = True

        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.sample

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")


class NCSNppModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor(batch_size * [10])

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64, 64, 64],
            "in_channels": 3,
            "layers_per_block": 1,
            "out_channels": 3,
            "time_embedding_type": "fourier",
            "norm_eps": 1e-6,
            "mid_block_scale_factor": math.sqrt(2.0),
            "norm_num_groups": None,
            "down_block_types": [
                "SkipDownBlock2D",
                "AttnSkipDownBlock2D",
                "SkipDownBlock2D",
                "SkipDownBlock2D",
            ],
            "up_block_types": [
                "SkipUpBlock2D",
                "SkipUpBlock2D",
                "AttnSkipUpBlock2D",
                "SkipUpBlock2D",
            ],
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @slow
    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        inputs = self.dummy_input
        noise = floats_tensor((4, 3) + (256, 256))
        inputs["sample"] = noise
        image = model(**inputs)

        assert image is not None, "Make sure output is not None"

    @slow
    def test_output_pretrained_ve_mid(self):
        model = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256")

        paddle.seed(0)

        batch_size = 4
        num_channels = 3
        sizes = (256, 256)

        noise = paddle.ones((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor(batch_size * [1e-4])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten()

        expected_output_slice = paddle.to_tensor(
            [-4836.2231, -6487.1387, -3816.7969, -7964.9253, -10966.2842, -20043.6016, 8137.0571, 2340.3499, 544.6114]
        )

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-2))

    def test_output_pretrained_ve_large(self):
        model = UNet2DModel.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy-update")

        paddle.seed(0)

        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = paddle.ones((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor(batch_size * [1e-4])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten()

        expected_output_slice = paddle.to_tensor(
            [-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256]
        )

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-2))

    def test_forward_with_norm_groups(self):
        # not required for this model
        pass


@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_latents(self, seed=0, shape=(4, 4, 64, 64)):
        image = paddle.to_tensor(load_ppnlp_numpy(self.get_file_format(seed, shape)))
        return image

    def get_unet_model(self, model_id="CompVis/stable-diffusion-v1-4"):
        model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        model.eval()

        return model

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768)):
        hidden_states = paddle.to_tensor(load_ppnlp_numpy(self.get_file_format(seed, shape)))
        return hidden_states

    @parameterized.expand(
        [
            [33, 4, [-0.4424, 0.1510, -0.1937, 0.2118, 0.3746, -0.3957, 0.0160, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.2540, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
        ]
    )
    def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32")
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            [33, 4, [-0.4430, 0.1570, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.3430, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.1750]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
        ]
    )
    def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32")
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.4120, -0.5621, -0.0397, -1.0430]],
            [9, 1000, [0.1600, 0.7303, -1.0556, -0.3515, -0.7440, -1.2037, -1.8149, -1.8931]],
        ]
    )
    def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting")
        latents = self.get_latents(seed, shape=(4, 9, 64, 64))
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        with paddle.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == [4, 4, 64, 64]

        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32")
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)
