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

import paddle

from ppdiffusers import KDPM2AncestralDiscreteScheduler

from .test_schedulers import SchedulerCommonTest


class KDPM2AncestralDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2AncestralDiscreteScheduler,)
    num_inference_steps = 10

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [10, 50, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.00001, 0.0001, 0.001], [0.0002, 0.002, 0.02]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "scaled_linear"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_full_loop_no_noise(self):

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 13913.05566406) < 1e-2
        assert abs(result_mean.item() - 18.11595917) < 5e-3

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_full_loop_with_v_prediction(self):

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        generator = paddle.Generator().manual_seed(0)

        for i, t in enumerate(scheduler.timesteps):
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 327.80270386) < 1e-2
        assert abs(result_mean.item() - 0.42682642) < 1e-3

    def test_full_loop_device(self):

        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(self.num_inference_steps)
        generator = paddle.Generator().manual_seed(0)

        model = self.dummy_model()
        sample = self.dummy_sample_deter * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            sample = scheduler.scale_model_input(sample, t)

            model_output = model(sample, t)

            output = scheduler.step(model_output, t, sample, generator=generator)
            sample = output.prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))
        assert abs(result_sum.item() - 13913.05566406) < 1e-1
        assert abs(result_mean.item() - 18.11595917) < 1e-3
