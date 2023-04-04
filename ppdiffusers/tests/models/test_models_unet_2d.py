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

import gc
import math
import tracemalloc
import unittest

import paddle
from ppdiffusers_test.test_modeling_common import ModelTesterMixin

from ppdiffusers import UNet2DModel
from ppdiffusers.utils import floats_tensor, logging, paddle_all_close, slow

logger = logging.get_logger(__name__)


class Unet2DModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = 32, 32
        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])
        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

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
        sizes = 32, 32
        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])
        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 4, 32, 32

    @property
    def output_shape(self):
        return 4, 32, 32

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

    def test_from_pretrained_accelerate(self):
        model, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        image = model(**self.dummy_input).sample
        assert image is not None, "Make sure output is not None"

    def test_from_pretrained_accelerate_wont_change_results(self):
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_accelerate
        model_accelerate.eval()
        noise = paddle.randn(
            shape=[
                1,
                model_accelerate.config.in_channels,
                model_accelerate.config.sample_size,
                model_accelerate.config.sample_size,
            ],
            generator=paddle.Generator().manual_seed(0),
        )
        time_step = paddle.to_tensor([10] * noise.shape[0])
        arr_accelerate = model_accelerate(noise, time_step)["sample"]
        del model_accelerate
        paddle.device.cuda.empty_cache()
        gc.collect()
        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update",
            output_loading_info=True,
        )
        model_normal_load.eval()
        arr_normal_load = model_normal_load(noise, time_step)["sample"]
        assert paddle_all_close(arr_accelerate, arr_normal_load, rtol=0.001)

    def test_memory_footprint_gets_reduced(self):
        paddle.device.cuda.empty_cache()
        gc.collect()
        tracemalloc.start()
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        model_accelerate
        model_accelerate.eval()
        _, peak_accelerate = tracemalloc.get_traced_memory()
        del model_accelerate
        paddle.device.cuda.empty_cache()
        gc.collect()
        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update",
            output_loading_info=True,
        )
        model_normal_load.eval()
        _, peak_normal = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak_accelerate < peak_normal

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()
        noise = paddle.randn(
            shape=[1, model.config.in_channels, model.config.sample_size, model.config.sample_size],
            generator=paddle.Generator().manual_seed(0),
        )
        time_step = paddle.to_tensor([10] * noise.shape[0])
        with paddle.no_grad():
            output = model(noise, time_step).sample
        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = paddle.to_tensor(
            [
                0.43855608,
                -10.29346752,
                -9.60953522,
                -8.39902020,
                -16.29206276,
                -13.07511997,
                -9.30383205,
                -13.69859409,
                -10.52999401,
            ]
        )
        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=0.001))


class NCSNppModelTests(ModelTesterMixin, unittest.TestCase):
    model_class = UNet2DModel

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3
        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor(batch_size * [10]).cast("int32")
        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64, 64, 64],
            "in_channels": 3,
            "layers_per_block": 1,
            "out_channels": 3,
            "time_embedding_type": "fourier",
            "norm_eps": 1e-06,
            "mid_block_scale_factor": math.sqrt(2.0),
            "norm_num_groups": None,
            "down_block_types": ["SkipDownBlock2D", "AttnSkipDownBlock2D", "SkipDownBlock2D", "SkipDownBlock2D"],
            "up_block_types": ["SkipUpBlock2D", "SkipUpBlock2D", "AttnSkipUpBlock2D", "SkipUpBlock2D"],
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
        sizes = 256, 256
        noise = paddle.ones(shape=(batch_size, num_channels, *sizes))
        time_step = paddle.to_tensor(batch_size * [0.0001])
        with paddle.no_grad():
            output = model(noise, time_step).sample
        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        expected_output_slice = paddle.to_tensor(
            [-4836.2231, -6487.1387, -3816.7969, -7964.9253, -10966.2842, -20043.6016, 8137.0571, 2340.3499, 544.6114]
        )
        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=0.01))

    def test_output_pretrained_ve_large(self):
        model = UNet2DModel.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy-update")
        paddle.seed(0)
        batch_size = 4
        num_channels = 3
        sizes = 32, 32
        noise = paddle.ones(shape=(batch_size, num_channels, *sizes))
        time_step = paddle.to_tensor(batch_size * [0.0001])
        with paddle.no_grad():
            output = model(noise, time_step).sample
        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        expected_output_slice = paddle.to_tensor(
            [-0.0325, -0.09, -0.0869, -0.0332, -0.0725, -0.027, -0.0101, 0.0227, 0.0256]
        )
        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=0.01))

    def test_forward_with_norm_groups(self):
        pass
