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

import lora_helper

from ppdiffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

lora_helper

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

# 我们加载safetensor版本的权重

# https://civitai.com/models/6779/arknights-texas-the-omertosa
lora_outputs_path = "xarknightsTexasThe_v10.safetensors"

# 加载之前的模型
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.apply_lora(lora_outputs_path)

prompt = "A photo of sks dog in a bucket"
negative_prompt = ""
guidance_scale = 8
num_inference_steps = 25
height = 512
width = 512

img = pipe(
    prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,
).images[0]
img.save("demo.png")
