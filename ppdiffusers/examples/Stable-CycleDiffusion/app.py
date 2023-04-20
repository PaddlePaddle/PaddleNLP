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

import abc
import os
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import paddle
import paddle.nn.functional as F
import ptp_utils
import seq_aligner
from PIL import Image

from ppdiffusers import CycleDiffusionPipeline, DDIMScheduler

LOW_RESOURCE = False
MAX_NUM_WORDS = 77

paddle_dtype = paddle.float32  # paddle.float32
model_id_or_path = "CompVis/stable-diffusion-v1-4"
device_print = "GPU 🔥"
device = "gpu"

pipe = CycleDiffusionPipeline.from_pretrained(
    model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"), paddle_dtype=paddle_dtype
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
tokenizer = pipe.tokenizer


class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape([self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS]) for item in maps]
        maps = paddle.concat(maps, axis=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = mask > self.threshold
        mask = (mask[:1] + mask[1:]).cast(x_t.dtype)
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words, threshold=0.3):
        alpha_layers = paddle.zeros([len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS])
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.cast(paddle_dtype)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # h = attn.shape[0]
                # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                attn[1:] = self.forward(attn[1:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand([att_replace.shape[0], *attn_base.shape])
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_replace_new = (
                    self.replace_cross_attention(attn_base, attn_repalce) * alpha_words
                    + (1 - alpha_words) * attn_repalce
                )
                attn[1:] = attn_replace_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
    ):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer
        ).cast(paddle_dtype)
        if type(self_replace_steps) is float or type(self_replace_steps) is int:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        return paddle.einsum("hpw,bwn->bhpn", attn_base, self.mapper)

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionReplace, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend
        )
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).cast(paddle_dtype)


class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        # a.shape [8, 4096, 77]
        # b.shape [1, 77]
        # pt: a[:, :, b].shape = torch.Size([8, 4096, 1, 77])
        # pd: a.take_along_axis(b.unsqueeze(0), axis=-1).unsqueeze(-2)

        attn_base_replace = (
            attn_base.take_along_axis(self.mapper.unsqueeze(0), axis=-1).unsqueeze(-2).transpose([2, 0, 1, 3])
        )
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
    ):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        alphas = alphas.cast(paddle_dtype)
        self.alphas = alphas.reshape([alphas.shape[0], 1, 1, alphas.shape[1]])


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = paddle.ones([len(values), 77])
    values = paddle.to_tensor(values, dtype=paddle_dtype)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def inference(
    source_prompt,
    target_prompt,
    source_guidance_scale=1,
    guidance_scale=5,
    num_inference_steps=100,
    width=512,
    height=512,
    seed=0,
    img=None,
    strength=0.7,
    cross_attention_control="None",
    cross_replace_steps=0.8,
    self_replace_steps=0.4,
):

    paddle.seed(seed)

    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    # make sure dtype is float
    source_guidance_scale = float(source_guidance_scale)
    guidance_scale = float(guidance_scale)
    strength = float(strength)
    self_replace_steps = float(self_replace_steps)
    cross_replace_steps = float(cross_replace_steps)

    # create the CAC controller.
    if cross_attention_control == "Replace":
        controller = AttentionReplace(
            [source_prompt, target_prompt],
            num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
        )
        ptp_utils.register_attention_control(pipe, controller)
    elif cross_attention_control == "Refine":
        controller = AttentionRefine(
            [source_prompt, target_prompt],
            num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
        )
        ptp_utils.register_attention_control(pipe, controller)
    elif cross_attention_control == "None":
        controller = EmptyControl()
        ptp_utils.register_attention_control(pipe, controller)
    else:
        raise ValueError("Unknown cross_attention_control: {}".format(cross_attention_control))

    with paddle.amp.auto_cast(True, level="O2"):
        results = pipe(
            prompt=target_prompt,
            source_prompt=source_prompt,
            image=img,
            num_inference_steps=num_inference_steps,
            eta=0.1,
            strength=strength,
            guidance_scale=guidance_scale,
            source_guidance_scale=source_guidance_scale,
        )
    if pipe.safety_checker is None:
        return results.images[0]
    else:
        return replace_nsfw_images(results)


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("images/nsfw.png")
    return results.images[0]


css = """.cycle-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.cycle-diffusion-div div h1{font-weight:900;margin-bottom:7px}.cycle-diffusion-div p{margin-bottom:10px;font-size:94%}.cycle-diffusion-div p a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""
with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
            <div class="cycle-diffusion-div">
              <div>
                <h1>CycleDiffusion with Stable Diffusion</h1>
              </div>
              <p>
                Demo for CycleDiffusion with Stable Diffusion. <br>
                CycleDiffusion (<a href="https://arxiv.org/abs/2210.05559">📄 Paper link</a> | <a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/cycle_diffusion">🧨 Pipeline doc</a>) is an image-to-image translation method that supports stochastic samplers for diffusion models. <br>
                We also support the combination of CycleDiffusion and Cross Attention Control (CAC | <a href="https://arxiv.org/abs/2208.01626">📄 Paper link</a>). CAC is a technique to transfer the attention map from the source prompt to the target prompt. <br>
              </p>
              <p>
              <b>Quick start</b>: <br>
              1. Click one row of Examples at the end of this page. It will fill all inputs needed. <br>
              2. Click the "Run CycleDiffusion" button. <br>
              </p>
            </div>
        """
    )
    with gr.Accordion("See Details", open=False):
        gr.HTML(
            """
            <div class="cycle-diffusion-div">
              <p>
                <b>How to use:</b> <br>
                1. Upload an image. <br>
                2. Enter the source and target prompts. <br>
                3. Select the source guidance scale (for "encoding") and the target guidance scale (for "decoding"). <br>
                4. Select the strength (smaller strength means better content preservation). <br>
                5 (optional). Configurate Cross Attention Control options (e.g., CAC type, cross replace steps, self replace steps). <br>
                6 (optional). Configurate other options (e.g., image size, inference steps, random seed). <br>
                7. Click the "Run CycleDiffusion" button. <br>
              </p>
              <p>
                <b>Notes:</b> <br>
                1. CycleDiffusion is likely to fail when drastic changes are intended (e.g., changing a large black car to red). <br>
                2. The value of strength can be set larger when CAC is used. <br>
                3. If CAC type is "Replace", the source and target prompts should differ in only one token; otherwise, an error will be raised. This is why we deliberately make some grammar mistakes in Examples.<br>
                4. If CAC type is "Refine", the source prompt be a subsequence of the target prompt; otherwise, an error will be raised. <br>
              </p>
              <p>
              <b>Runtimes:</b> <br>
              1. 20s on A10G. <br>
              </p>
            </div>
        """
        )
    with gr.Row():

        with gr.Column(scale=55):
            with gr.Group():

                img = gr.Image(label="Input image", height=512, tool="editor", type="pil")

                image_out = gr.Image(label="Output image", height=512)
                # gallery = gr.Gallery(
                #     label="Generated images", show_label=False, elem_id="gallery"
                # ).style(grid=[1], height="auto")

        with gr.Column(scale=45):
            with gr.Tab("Edit options"):
                with gr.Group():
                    with gr.Row():
                        source_prompt = gr.Textbox(
                            label="Source prompt", placeholder="Source prompt describes the input image"
                        )
                        source_guidance_scale = gr.Slider(
                            label="Source guidance scale", value=1, minimum=1, maximum=10
                        )
                    with gr.Row():
                        target_prompt = gr.Textbox(
                            label="Target prompt", placeholder="Target prompt describes the output image"
                        )
                        guidance_scale = gr.Slider(label="Target guidance scale", value=5, minimum=1, maximum=10)
                    with gr.Row():
                        strength = gr.Slider(label="Strength", value=0.7, minimum=0.5, maximum=1, step=0.01)
                    with gr.Row():
                        generate1 = gr.Button(value="Run CycleDiffusion")

            with gr.Tab("CAC options"):
                with gr.Group():
                    with gr.Row():
                        cross_attention_control = gr.Radio(
                            label="CAC type", choices=["None", "Replace", "Refine"], value="None"
                        )
                    with gr.Row():
                        # If not "None", the following two parameters will be used.
                        cross_replace_steps = gr.Slider(
                            label="Cross replace steps", value=0.8, minimum=0.0, maximum=1, step=0.01
                        )
                        self_replace_steps = gr.Slider(
                            label="Self replace steps", value=0.4, minimum=0.0, maximum=1, step=0.01
                        )
                    with gr.Row():
                        generate2 = gr.Button(value="Run CycleDiffusion")

            with gr.Tab("Other options"):
                with gr.Group():
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Inference steps", value=100, minimum=25, maximum=500, step=1
                        )
                        width = gr.Slider(label="Width", value=512, minimum=512, maximum=1024, step=8)
                        height = gr.Slider(label="Height", value=512, minimum=512, maximum=1024, step=8)

                    with gr.Row():
                        seed = gr.Slider(0, 2147483647, label="Seed", value=0, step=1)
                    with gr.Row():
                        generate3 = gr.Button(value="Run CycleDiffusion")

    inputs = [
        source_prompt,
        target_prompt,
        source_guidance_scale,
        guidance_scale,
        num_inference_steps,
        width,
        height,
        seed,
        img,
        strength,
        cross_attention_control,
        cross_replace_steps,
        self_replace_steps,
    ]
    generate1.click(inference, inputs=inputs, outputs=image_out)
    generate2.click(inference, inputs=inputs, outputs=image_out)
    generate3.click(inference, inputs=inputs, outputs=image_out)

    ex = gr.Examples(
        [
            [
                "An astronaut riding a horse",
                "An astronaut riding an elephant",
                1,
                2,
                100,
                512,
                512,
                0,
                "images/astronaut_horse.png",
                0.8,
                "None",
                0,
                0,
            ],
            [
                "An astronaut riding a horse",
                "An astronaut riding a elephant",
                1,
                2,
                100,
                512,
                512,
                0,
                "images/astronaut_horse.png",
                0.9,
                "Replace",
                0.15,
                0.10,
            ],
            [
                "A black colored car.",
                "A blue colored car.",
                1,
                3,
                100,
                512,
                512,
                0,
                "images/black_car.png",
                0.85,
                "None",
                0,
                0,
            ],
            [
                "A black colored car.",
                "A blue colored car.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/black_car.png",
                0.95,
                "Replace",
                0.8,
                0.4,
            ],
            [
                "A black colored car.",
                "A red colored car.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/black_car.png",
                1,
                "Replace",
                0.8,
                0.4,
            ],
            [
                "An aerial view of autumn scene.",
                "An aerial view of winter scene.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/mausoleum.png",
                0.9,
                "None",
                0,
                0,
            ],
            [
                "An aerial view of autumn scene.",
                "An aerial view of winter scene.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/mausoleum.png",
                1,
                "Replace",
                0.8,
                0.4,
            ],
            [
                "A green apple and a black backpack on the floor.",
                "A red apple and a black backpack on the floor.",
                1,
                7,
                100,
                512,
                512,
                0,
                "images/apple_bag.png",
                0.9,
                "None",
                0,
                0,
            ],
            [
                "A green apple and a black backpack on the floor.",
                "A red apple and a black backpack on the floor.",
                1,
                7,
                100,
                512,
                512,
                0,
                "images/apple_bag.png",
                0.9,
                "Replace",
                0.8,
                0.4,
            ],
            [
                "A hotel room with red flowers on the bed.",
                "A hotel room with a cat sitting on the bed.",
                1,
                4,
                100,
                512,
                512,
                0,
                "images/flower_hotel.png",
                0.8,
                "None",
                0,
                0,
            ],
            [
                "A hotel room with red flowers on the bed.",
                "A hotel room with blue flowers on the bed.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/flower_hotel.png",
                0.95,
                "None",
                0,
                0,
            ],
            [
                "A green apple and a black backpack on the floor.",
                "Two green apples and a black backpack on the floor.",
                1,
                5,
                100,
                512,
                512,
                0,
                "images/apple_bag.png",
                0.89,
                "None",
                0,
                0,
            ],
        ],
        [
            source_prompt,
            target_prompt,
            source_guidance_scale,
            guidance_scale,
            num_inference_steps,
            width,
            height,
            seed,
            img,
            strength,
            cross_attention_control,
            cross_replace_steps,
            self_replace_steps,
        ],
        image_out,
        inference,
        cache_examples=True,
    )

    gr.Markdown(
        """
      Space built with PPDiffusers 🧨 by PaddleNLP.
      [![Twitter Follow](https://img.shields.io/twitter/follow/ChenHenryWu?style=social)](https://twitter.com/ChenHenryWu)
      """
    )

demo.launch(debug=True, share=True, server_name="0.0.0.0", server_port=8581)
