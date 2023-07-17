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

from ppdiffusers import DDPMScheduler

from .test_schedulers import SchedulerCommonTest


class DDPMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_thresholding(self):
        self.check_over_configs(thresholding=False)
        for threshold in [0.5, 1.0, 2.0]:
            for prediction_type in ["epsilon", "sample", "v_prediction"]:
                self.check_over_configs(
                    thresholding=True,
                    prediction_type=prediction_type,
                    sample_max_value=threshold,
                )

    def test_prediction_type(self):
        for prediction_type in ["epsilon", "sample", "v_prediction"]:
            self.check_over_configs(prediction_type=prediction_type)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert paddle.sum(paddle.abs(scheduler._get_variance(0) - 0.0)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(487) - 0.00979)) < 1e-5
        assert paddle.sum(paddle.abs(scheduler._get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 261.0068359375) < 1e-2
        assert abs(result_mean.item() - 0.33985263109207153) < 1e-3

    def test_full_loop_with_v_prediction(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(prediction_type="v_prediction")
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = paddle.Generator().manual_seed(0)

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            pred_prev_sample = scheduler.step(residual, t, sample, generator=generator).prev_sample

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = paddle.sum(paddle.abs(sample))
        result_mean = paddle.mean(paddle.abs(sample))

        assert abs(result_sum.item() - 202.7893524169922) < 1e-2
        assert abs(result_mean.item() - 0.26404863595962524) < 1e-3

    def test_custom_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]

        scheduler.set_timesteps(timesteps=timesteps)

        scheduler_timesteps = scheduler.timesteps

        for i, timestep in enumerate(scheduler_timesteps):
            if i == len(timesteps) - 1:
                expected_prev_t = -1
            else:
                expected_prev_t = timesteps[i + 1]

            prev_t = scheduler.previous_timestep(timestep)
            prev_t = prev_t.item()

            self.assertEqual(prev_t, expected_prev_t)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 51, 0]

        with self.assertRaises(ValueError, msg="`custom_timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `custom_timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)
