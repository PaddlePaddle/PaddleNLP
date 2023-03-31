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

from ppdiffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from ppdiffusers.training_utils import set_seed
from ppdiffusers.utils.testing_utils import slow


class TrainingTests(unittest.TestCase):
    def get_model_optimizer(self, resolution=32):
        set_seed(0)
        model = UNet2DModel(sample_size=resolution, in_channels=3, out_channels=3)
        optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=0.0001)
        return model, optimizer

    @slow
    def test_training_step_equality(self):
        ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", clip_sample=True
        )
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear", clip_sample=True
        )
        assert ddpm_scheduler.config.num_train_timesteps == ddim_scheduler.config.num_train_timesteps
        set_seed(0)
        clean_images = [paddle.randn(shape=(4, 3, 32, 32)).clip(min=-1, max=1) for _ in range(4)]
        noise = [paddle.randn(shape=(4, 3, 32, 32)) for _ in range(4)]
        timesteps = [paddle.randint(0, 1000, (4,)).astype(dtype="int64") for _ in range(4)]
        model, optimizer = self.get_model_optimizer(resolution=32)
        model.train()
        for i in range(4):
            optimizer.clear_grad()
            ddpm_noisy_images = ddpm_scheduler.add_noise(clean_images[i], noise[i], timesteps[i])
            ddpm_noise_pred = model(ddpm_noisy_images, timesteps[i]).sample
            loss = paddle.nn.functional.mse_loss(input=ddpm_noise_pred, label=noise[i])
            loss.backward()
            optimizer.step()
        del model, optimizer
        model, optimizer = self.get_model_optimizer(resolution=32)
        model.train()
        for i in range(4):
            optimizer.clear_grad()
            ddim_noisy_images = ddim_scheduler.add_noise(clean_images[i], noise[i], timesteps[i])
            ddim_noise_pred = model(ddim_noisy_images, timesteps[i]).sample
            loss = paddle.nn.functional.mse_loss(input=ddim_noise_pred, label=noise[i])
            loss.backward()
            optimizer.step()
        del model, optimizer
        self.assertTrue(paddle.allclose(ddpm_noisy_images, ddim_noisy_images, atol=1e-05))
        self.assertTrue(paddle.allclose(ddpm_noise_pred, ddim_noise_pred, atol=1e-04))
