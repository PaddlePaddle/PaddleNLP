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

import random

import gradio as gr
import paddle
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image

from paddlenlp.trainer import set_seed as seed_everything
from ppdiffusers import ControlNetModel, StableDiffusionControlNetPipeline

apply_canny = CannyDetector()

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
)


def process(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    with paddle.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = paddle.to_tensor(detected_map.copy(), dtype=paddle.float32) / 255.0
        control = control.unsqueeze(0).transpose([0, 3, 1, 2])

        control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        results = []
        for _ in range(num_samples):
            img = pipe(
                prompt + ", " + a_prompt,
                negative_prompt=n_prompt,
                image=control,
                num_inference_steps=ddim_steps,
                height=H,
                width=W,
                eta=eta,
                controlnet_conditioning_scale=control_scales,
                guidance_scale=scale,
            ).images[0]
            results.append(img)

    return [255 - detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value="best quality, extremely detailed")
                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                )
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(
                grid=2, height="auto"
            )
    ips = [
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
    ]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name="0.0.0.0", server_port=8513)
