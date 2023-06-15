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
from clip_guided_stable_diffusion import CLIPGuidedStableDiffusion
from IPython.display import display
from PIL import Image

from paddlenlp.transformers import CLIPFeatureExtractor, CLIPModel
from ppdiffusers import LMSDiscreteScheduler, StableDiffusionPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def create_clip_guided_pipeline(
    model_id="CompVis/stable-diffusion-v1-4", clip_model_id="openai/clip-vit-large-patch14", scheduler="plms"
):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)

    if scheduler == "lms":
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    else:
        scheduler = pipeline.scheduler

    clip_model = CLIPModel.from_pretrained(clip_model_id)
    feature_extractor = CLIPFeatureExtractor()

    guided_pipeline = CLIPGuidedStableDiffusion(
        unet=pipeline.unet,
        vae=pipeline.vae,
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        scheduler=scheduler,
        clip_model=clip_model,
        feature_extractor=feature_extractor,
    )

    return guided_pipeline


def infer(
    prompt,
    clip_prompt,
    num_return_images=1,
    num_images_per_prompt=1,
    num_inference_steps=50,
    clip_guidance_scale=100,
    guidance_scale=7.5,
    guided_pipeline=None,
    negative_prompt="",
    use_cutouts=True,
    num_cutouts=4,
    seed=None,
    unfreeze_unet=True,
    unfreeze_vae=True,
):
    clip_prompt = clip_prompt if clip_prompt.strip() != "" else None
    if unfreeze_unet:
        guided_pipeline.unfreeze_unet()
    else:
        guided_pipeline.freeze_unet()

    if unfreeze_vae:
        guided_pipeline.unfreeze_vae()
    else:
        guided_pipeline.freeze_vae()

    images = []
    for i in range(num_return_images):
        image = guided_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            clip_prompt=clip_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            clip_guidance_scale=clip_guidance_scale,
            num_cutouts=num_cutouts,
            use_cutouts=use_cutouts,
            seed=seed,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        images.extend(image)

    return image_grid(images, 1, len(images))


if __name__ == "__main__":
    prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"  # @param {type: "string"}
    # @markdown `clip_prompt` is optional, if you leave it blank the same prompt is sent to Stable Diffusion and CLIP
    clip_prompt = ""  # @param {type: "string"}
    negative_prompt = ""
    num_return_images = 1  # @param {type: "number"}
    num_images_per_prompt = 1  # @param {type: "number"}

    num_inference_steps = 50  # @param {type: "number"}
    guidance_scale = 7.5  # @param {type: "number"}
    clip_guidance_scale = 100  # @param {type: "number"}
    num_cutouts = 4  # @param {type: "number"}
    use_cutouts = False  # @param ["False", "True"]
    unfreeze_unet = False  # @param ["False", "True"]
    unfreeze_vae = False  # @param ["False", "True"]
    seed = 3788086447  # @param {type: "number"}

    model_id = "CompVis/stable-diffusion-v1-4"
    clip_model_id = "openai/clip-vit-base-patch32"  # @param ["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch14", "openai/clip-rn101", "openai/clip-rn50"] {allow-input: true}
    scheduler = "plms"  # @param ['plms', 'lms']
    guided_pipeline = create_clip_guided_pipeline(model_id, clip_model_id)

    with paddle.amp.auto_cast(True, level="O2"):
        grid_image = infer(
            prompt=prompt,
            negative_prompt=negative_prompt,
            clip_prompt=clip_prompt,
            num_return_images=num_return_images,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            clip_guidance_scale=clip_guidance_scale,
            guidance_scale=guidance_scale,
            guided_pipeline=guided_pipeline,
            use_cutouts=use_cutouts,
            num_cutouts=num_cutouts,
            seed=seed,
            unfreeze_unet=unfreeze_unet,
            unfreeze_vae=unfreeze_vae,
        )

    display(grid_image)
