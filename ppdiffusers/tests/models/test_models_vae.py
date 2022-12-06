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
import unittest

import paddle
from parameterized import parameterized
from test_modeling_common import ModelTesterMixin

from ppdiffusers import AutoencoderKL
from ppdiffusers.utils import floats_tensor, load_ppnlp_numpy, paddle_all_close, slow


class AutoencoderKLTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy")
        model.eval()

        generator = paddle.Generator().manual_seed(0)

        image = paddle.randn(
            [1, model.config.in_channels, model.config.sample_size, model.config.sample_size],
            generator=generator,
        )
        with paddle.no_grad():
            output = model(image, sample_posterior=True, generator=generator).sample

        output_slice = output[0, -1, -3:, -3:].flatten()

        # Since the VAE Gaussian prior's generator is seeded on the appropriate device,
        # the expected output slices are not the same for CPU and GPU.
        expected_output_slice = paddle.to_tensor(
            [
                -0.37055650,
                -0.33920342,
                0.08271024,
                -0.22468489,
                -0.04493487,
                0.02585240,
                -0.03970414,
                -0.08472210,
                -0.06026033,
            ]
        )

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-2))


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512)):
        image = paddle.to_tensor(load_ppnlp_numpy(self.get_file_format(seed, shape)))
        return image

    def get_sd_vae_model(self, model_id="CompVis/stable-diffusion-v1-4"):

        model = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
        )
        model.eval()

        return model

    def get_generator(self, seed=0):
        return paddle.Generator().manual_seed(seed)

    @parameterized.expand(
        [
            [33, [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824]],
            [47, [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089]],
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with paddle.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32")
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            [33, [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814]],
            [47, [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085]],
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)

        with paddle.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().cast("float32")
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.4990, -0.3720, -0.4925]],
        ]
    )
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with paddle.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten()
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with paddle.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)

        assert list(sample.shape) == [image.shape[0], 4] + [i // 8 for i in image.shape[2:]]

        output_slice = sample[0, -1, -3:, -3:].flatten()
        expected_output_slice = paddle.to_tensor(expected_slice)

        assert paddle_all_close(output_slice, expected_output_slice, atol=1e-3)
