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

from io import BytesIO

import paddle
import PIL
import requests

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPModel
from ppdiffusers import DiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


# Loading additional models
feature_extractor = CLIPFeatureExtractor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", paddle_dtype=paddle.float16)

mixing_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="clip_guided_images_mixing_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    paddle_dtype=paddle.float16,
)
mixing_pipeline.enable_attention_slicing()

# Pipline running
generator = paddle.Generator().manual_seed(17)


content_image = download_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop/clip_guided_images_mixing_stable_diffusion_images/boromir.jpg"
)
style_image = download_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop/clip_guided_images_mixing_stable_diffusion_images/gigachad.jpg"
)

pipe_images = mixing_pipeline(
    num_inference_steps=50,
    content_image=content_image,
    style_image=style_image,
    content_prompt="boromir",
    style_prompt="gigachad",
    noise_strength=0.65,
    slerp_latent_style_strength=0.9,
    slerp_prompt_style_strength=0.1,
    slerp_clip_image_style_strength=0.1,
    guidance_scale=9.0,
    batch_size=1,
    clip_guidance_scale=100,
    generator=generator,
).images

pipe_images[0].save("clip_guided_images_mixing_stable_diffusion.png")
