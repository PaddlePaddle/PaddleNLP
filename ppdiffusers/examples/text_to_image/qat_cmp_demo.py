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

import gradio as gr

from ppdiffusers import StableDiffusionPipeline

block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

pipe1 = StableDiffusionPipeline.from_pretrained("examples/text_to_image/sd-pokemon-model-bs4/").to("cpu")
pipe2 = StableDiffusionPipeline.from_pretrained(
    "examples/text_to_image/sd-pokemon-model-int8-onnx-nopact-distil02/final", use_qat=2
)

num_samples = 4
num_inference_steps = 50
seed = None  # 123


def on_gpu(pipeline):
    return "cpu" not in str(next(pipeline.unet.named_parameters())[1].place)


def infer(prompt):
    if on_gpu(pipe1):
        images1 = pipe1([prompt] * num_samples).images
        pipe1.to("cpu")
        pipe2.to("gpu")
        images2 = pipe2([prompt] * num_samples).images
    else:
        images2 = pipe2([prompt] * num_samples).images
        pipe2.to("cpu")
        pipe1.to("gpu")
        images1 = pipe1([prompt] * num_samples).images
    return images1 + images2


with block as demo:
    gr.Markdown("<h1><center>PaddleNLP version of Stable Diffusion</center></h1>")

    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(label="Enter your prompt", show_label=False, max_lines=1).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[num_samples], height="auto")
        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)

    gr.Markdown(
        """___
        <p style='text-align: center'>
        Created by https://huggingface.co/CompVis/stable-diffusion-v1-4
        <br/>
        </p>"""
    )

demo.launch(debug=True, server_port=8235, server_name="0.0.0.0", share=True)
