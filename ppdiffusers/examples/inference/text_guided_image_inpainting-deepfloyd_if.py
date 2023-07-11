# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ppdiffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline
from ppdiffusers.utils import load_image, pd_to_pil

url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
original_image = load_image(url)

url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
mask_image = load_image(url)

pipe = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", paddle_dtype=paddle.float16)
pipe.enable_xformers_memory_efficient_attention()
prompt = "blue sunglasses"
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

image = pipe(
    image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images
pipe.to(paddle_device="cpu")
# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("./text_guided_image_inpainting-deepfloyd_if-if_stage_I.png")

super_res_1_pipe = IFInpaintingSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", paddle_dtype=paddle.float16
)
super_res_1_pipe.enable_xformers_memory_efficient_attention()

image = super_res_1_pipe(
    image=image,
    mask_image=mask_image,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
).images
image[0].save("./text_guided_image_inpainting-deepfloyd_if-if_stage_II.png")
