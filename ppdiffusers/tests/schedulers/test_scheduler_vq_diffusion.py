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
import paddle.nn.functional as F

from ppdiffusers import VQDiffusionScheduler

from .test_schedulers import SchedulerCommonTest


class VQDiffusionSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (VQDiffusionScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_vec_classes": 4097,
            "num_train_timesteps": 100,
        }

        config.update(**kwargs)
        return config

    def dummy_sample(self, num_vec_classes):
        batch_size = 4
        height = 8
        width = 8

        sample = paddle.randint(0, num_vec_classes, (batch_size, height * width))

        return sample

    @property
    def dummy_sample_deter(self):
        assert False

    def dummy_model(self, num_vec_classes):
        def model(sample, t, *args):
            batch_size, num_latent_pixels = sample.shape
            logits = paddle.rand((batch_size, num_vec_classes - 1, num_latent_pixels))
            return_value = F.log_softmax(logits.cast("float64"), axis=1).cast("float32")
            return return_value

        return model

    def test_timesteps(self):
        for timesteps in [2, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_num_vec_classes(self):
        for num_vec_classes in [5, 100, 1000, 4000]:
            self.check_over_configs(num_vec_classes=num_vec_classes)

    def test_time_indices(self):
        for t in [0, 50, 99]:
            self.check_over_forward(time_step=t)

    def test_add_noise_device(self):
        pass
