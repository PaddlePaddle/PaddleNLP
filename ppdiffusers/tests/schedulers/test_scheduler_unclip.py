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

from ppdiffusers import UnCLIPScheduler

from .test_schedulers import SchedulerCommonTest


# UnCLIPScheduler is a modified DDPMScheduler with a subset of the configuration.
class UnCLIPSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (UnCLIPScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "variance_type": "fixed_small_log",
            "clip_sample": True,
            "clip_sample_range": 1.0,
            "prediction_type": "epsilon",
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_variance_type(self):
        for variance in ["fixed_small_log", "learned_range"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_clip_sample_range(self):
        for clip_sample_range in [1, 5, 10, 20]:
            self.check_over_configs(clip_sample_range=clip_sample_range)

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for time_step in [0, 500, 999]:
            for prev_timestep in [None, 5, 100, 250, 500, 750]:
                if prev_timestep is not None and prev_timestep >= time_step:
                    continue

                self.check_over_forward(time_step=time_step, prev_timestep=prev_timestep)

    def test_variance_fixed_small_log(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(variance_type="fixed_small_log")
        scheduler = scheduler_class(**scheduler_config)

        assert paddle.sum(paddle.abs(scheduler._get_variance(0) - 1.0000e-10)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(487) - 0.0549625)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(999) - 0.9994987)) < 1e-5

    def test_variance_learned_range(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(variance_type="learned_range")
        scheduler = scheduler_class(**scheduler_config)

        predicted_variance = 0.5

        assert scheduler._get_variance(1, predicted_variance=predicted_variance) - -10.1712790 < 1e-5
        assert scheduler._get_variance(487, predicted_variance=predicted_variance) - -5.7998052 < 1e-5
        assert scheduler._get_variance(999, predicted_variance=predicted_variance) - -0.0010011 < 1e-5

    def test_full_loop(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = scheduler.timesteps

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for i, t in enumerate(timesteps):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 255.86759949) < 1e-2
        assert abs(result_mean.item() - 0.33316097) < 1e-3

    def test_full_loop_skip_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        scheduler.set_timesteps(25)

        timesteps = scheduler.timesteps

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for i, t in enumerate(timesteps):
            # 1. predict noise residual
            residual = model(sample, t)

            if i + 1 == timesteps.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = timesteps[i + 1]

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(
                residual, t, sample, prev_timestep=prev_timestep, generator=generator
            ).prev_sample

            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 249.76672363) < 1e-2
        assert abs(result_mean.item() - 0.32521713) < 1e-3

    def test_trained_betas(self):
        pass

    def test_add_noise_device(self):
        pass
