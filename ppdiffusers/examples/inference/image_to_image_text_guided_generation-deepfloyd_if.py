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

from ppdiffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline
from ppdiffusers.utils import load_image, pd_to_pil

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
original_image = load_image(url)
original_image = original_image.resize((768, 512))

pipe = IFImg2ImgPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    variant="fp16",
    paddle_dtype=paddle.float16,
)
pipe.enable_xformers_memory_efficient_attention()
prompt = "A fantasy landscape in style minecraft"
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

image = pipe(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images
pipe.to(paddle_device="cpu")

# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("./image_to_image_text_guided_generation-deepfloyd_if-if_stage_I.png")

super_res_1_pipe = IFImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0",
    text_encoder=None,
    variant="fp16",
    paddle_dtype=paddle.float16,
)
super_res_1_pipe.enable_xformers_memory_efficient_attention()

image = super_res_1_pipe(
    image=image,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
).images
image[0].save("./image_to_image_text_guided_generation-deepfloyd_if-if_stage_II.png")
