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

from ppdiffusers import DiffusionPipeline, IFPipeline, IFSuperResolutionPipeline
from ppdiffusers.utils import pd_to_pil

# Stage 1: generate images
pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", paddle_dtype=paddle.float16)
pipe.enable_xformers_memory_efficient_attention()
prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images

# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("text_to_image_generation-deepfloyd_if-result-if_stage_I.png")
# save gpu memory
pipe.to(paddle_device="cpu")

# Stage 2: super resolution stage1
super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", paddle_dtype=paddle.float16
)
super_res_1_pipe.enable_xformers_memory_efficient_attention()

image = super_res_1_pipe(
    image=image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images
# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("text_to_image_generation-deepfloyd_if-result-if_stage_II.png")
# save gpu memory
super_res_1_pipe.to(paddle_device="cpu")

# Stage 3: super resolution stage2
super_res_2_pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", paddle_dtype=paddle.float16
)
super_res_2_pipe.enable_xformers_memory_efficient_attention()

image = super_res_2_pipe(
    prompt=prompt,
    image=image,
).images
image[0].save("text_to_image_generation-deepfloyd_if-result-if_stage_III.png")
