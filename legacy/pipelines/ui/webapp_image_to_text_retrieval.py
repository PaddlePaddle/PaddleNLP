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
from io import BytesIO

import gradio as gr
from PIL import Image, ImageFile
from utils import image_to_text_search

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--serving_port", default=8502, type=int, help="Port for the serving.")
args = parser.parse_args()
# yapf: enable


def pil_base64(image, img_format="JPEG"):
    Image.MAX_IMAGE_PIXELS = 1000000000
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_buffer = BytesIO()
    image.save(img_buffer, format=img_format)
    byte_data = img_buffer.getvalue()
    return byte_data


def infer(query, top_k_retriever):
    image = pil_base64(query)
    response = image_to_text_search(image, top_k_retriever=top_k_retriever)
    texts = [[item["content"]] for item in response["documents"]]
    return texts


def main():
    block = gr.Blocks()
    title = "<h1 align='center'>ERNIE VIL 2.0 图到文搜索应用</h1>"
    description = "本项目为ERNIE-ViL 2.0等CLIP中文版模型的DEMO，可用于图文检索和图像、文本的表征提取，应用于文图搜索、文图推荐、零样本分类、视频检索等应用场景。"

    with block:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    image = gr.components.Image(label="图片", type="pil", elem_id=1)
                top_k = gr.components.Slider(minimum=0, maximum=50, step=1, value=8, label="返回文本数", elem_id=2)
                btn = gr.Button(
                    "搜索",
                )
            with gr.Column(scale=100):
                out = gr.Dataframe(
                    headers=["content"],
                    datatype=["str"],
                    label="搜索结果为：",
                )
        inputs = [image, top_k]
        btn.click(fn=infer, inputs=inputs, outputs=out)
    return block


if __name__ == "__main__":
    block = main()
    block.launch(server_name="0.0.0.0", server_port=args.serving_port, share=False)
