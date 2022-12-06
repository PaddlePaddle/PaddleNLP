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

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.enable_attention_slicing()
block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

num_samples = 2


def infer(prompt):
    images = pipe([prompt] * num_samples).images
    return images


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

        gallery = gr.Gallery(label="Generated images", show_label=False).style(grid=[2], height="auto")
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
