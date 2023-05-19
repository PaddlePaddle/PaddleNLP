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
import paddle

from ppdiffusers import DiffusionPipeline


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()

        self.register_modules(unet=unet, scheduler=scheduler)

    def __call__(self):
        image = paddle.randn(
            (1, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
        )
        timestep = 1

        model_output = self.unet(image, timestep).sample
        scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

        return scheduler_output
