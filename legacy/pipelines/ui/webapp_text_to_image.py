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

import argparse

import gradio as gr
from utils import text_to_image_search

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--serving_port", default=8502, type=int, help="Port for the serving.")
args = parser.parse_args()
# yapf: enable


def infer(text_prompt, top_k_images, Size, style):
    results, raw_json = text_to_image_search(text_prompt, resolution=Size, top_k_images=top_k_images, style=style)
    return results


def main():
    block = gr.Blocks()

    with block:
        with gr.Group():
            with gr.Box():
                with gr.Row().style(mobile_collapse=False, equal_height=True):
                    text_prompt = gr.Textbox(
                        label="Enter your prompt",
                        value="宁静的小镇",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    btn = gr.Button("开始生成").style(
                        margin=False,
                        rounded=(False, True, True, False),
                    )
            gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(
                grid=[2], height="auto"
            )

            advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")

            with gr.Row(elem_id="advanced-options"):
                top_k_images = gr.Slider(label="Images", minimum=1, maximum=50, value=5, step=1)
                style = gr.Radio(
                    label="Style",
                    value="古风",
                    choices=[
                        "古风",
                        "油画",
                        "卡通画",
                        "二次元",
                        "水彩画",
                        "浮世绘",
                        "蒸汽波艺术",
                        "low poly",
                        "像素风格",
                        "概念艺术",
                        "未来主义",
                        "赛博朋克",
                        "写实风格",
                        "洛丽塔风格",
                        "巴洛克风格",
                        "超现实主义",
                        "探索无限",
                    ],
                )
                Size = gr.Radio(label="Size", value="1024*1024", choices=["1024*1024", "1024*1536", "1536*1024"])

            text_prompt.submit(infer, inputs=[text_prompt, top_k_images, Size, style], outputs=gallery)
            btn.click(infer, inputs=[text_prompt, top_k_images, Size, style], outputs=gallery)
            advanced_button.click(
                None,
                [],
                text_prompt,
            )
    return block


if __name__ == "__main__":
    block = main()
    block.launch(server_name="0.0.0.0", server_port=args.serving_port, share=False)
