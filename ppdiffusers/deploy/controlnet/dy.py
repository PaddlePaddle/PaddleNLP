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

import time

import cv2
import numpy as np
from PIL import Image

from ppdiffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
)
from ppdiffusers.utils import load_image

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "takuma104/control_sd15_canny",
    safety_checker=None,
)

benchmark_steps = 10
inference_steps = 50
image_path = "dongtaitu.png"
image = load_image("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny.png")

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

prompt = "bird"
scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
pipe.scheduler = scheduler
out_image = pipe(prompt=prompt, image=canny_image, num_inference_steps=10).images[0]

time_costs = []
print(f"Run the controlnet pipeline {benchmark_steps} times to test the performance.")

for step in range(benchmark_steps):
    start = time.time()
    images = pipe(prompt=prompt, image=canny_image, num_inference_steps=inference_steps).images
    latency = time.time() - start
    time_costs += [latency]
    print(f"No {step:3d} time cost: {latency:2f} s")
print(
    f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
    f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
)

images[0].save(image_path)
print(f"Image saved in {image_path}!")
