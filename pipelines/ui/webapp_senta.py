# -*- coding: UTF-8 -*-
# Copyright 2022 The Impira Team and the HuggingFace Team.
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
import os

import gradio as gr
import requests

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--serving_name', default="0.0.0.0", help="Serving ip.")
parser.add_argument("--serving_port", default=8891, type=int, help="Serving port.")
args = parser.parse_args()
# yapf: enable


def process_upload(file):
    if file:
        return [gr.update(label="uploaded"), gr.update(value=file.name), gr.update(visible=False, value=None)]
    else:
        return [gr.update(label="Select a file", value=None), None, gr.update(visible=False, value=None)]


def process_file(file_path):
    if not file_path:
        return (
            [gr.update(visible=False)]
            + [None] * 12
            + [gr.update(visible=True, value="Warm Note: upload file firstly")]
        )

    url = f"http://{args.serving_name}:{args.serving_port}/senta_file"
    save_path = os.path.join(os.path.dirname(file_path), "senta_" + os.path.basename(file_path))
    r = requests.post(url, json={"meta": {"file_path": file_path, "save_path": save_path}})
    response = r.json()
    results = response["img_dict"]
    components = [
        "aspect_wc",
        "aspect_hist",
        "opinion_wc",
        "opinion_hist",
        "aspect_opinion_wc",
        "aspect_opinion_hist",
        "aspect_opinion_wc_pos",
        "aspect_opinion_hist_pos",
        "aspect_opinion_wc_neg",
        "aspect_opinion_hist_neg",
        "aspect_sentiment_wc",
        "aspect_sentiment_hist",
    ]
    outputs = [gr.update(visible=True)]
    for component in components:
        if component in results:
            outputs.append(results[component])
        else:
            outputs.append(None)
    outputs.append(gr.update(visible=False, value=None))
    return outputs


def reset_click():
    return [
        gr.update(value=None, label="Select a file"),
        None,
        gr.update(visible=False, value=None),
        gr.update(visible=False),
    ]


with gr.Blocks() as demo:
    file_path = gr.Textbox(visible=False)
    gr.Markdown(value="# Sentiment Analysis Application\n----")
    upload_file = gr.File(label="Select a file", interactive=True, elem_id="file-upload-box")
    with gr.Row():
        reset_btn = gr.Button("Reset")
        file_btn = gr.Button("Submit")

    # define something with exceptional situation
    msg_box = gr.Markdown(visible=False, interactive=False)

    # show sentiment analysis with UI for batch processing
    with gr.Column(visible=False) as show:
        gr.Markdown(value="----")
        gr.Markdown(value="# Sentiment Analysis Show")
        gr.Markdown("## 1. 属性分析\n通过属性信息，可以查看客户对于产品/服务的重点关注方面. ")
        with gr.Row(equal_height=True):
            aspect_wc = gr.Image()
            aspect_hist = gr.Image()
        gr.Markdown("## 2. 观点分析\n通过观点信息，可以查看客户对于产品/服务整体的直观印象。")
        with gr.Row(equal_height=True):
            opinion_wc = gr.Image()
            opinion_hist = gr.Image()
        gr.Markdown("## 3. 属性 + 观点分析\n结合属性和观点两者信息，可以更加具体的展现客户对于产品/服务的详细观点，分析某个属性的优劣，从而能够帮助商家更有针对性地改善或提高自己的产品/服务质量。")
        with gr.Column(equal_height=True):
            gr.Markdown("### 3.1 全部属性+观点的内容分析")
            with gr.Row():
                aspect_opinion_wc = gr.Image()
                aspect_opinion_hist = gr.Image()
            gr.Markdown("### 3.2 正向属性+观点的内容分析")
            with gr.Row():
                aspect_opinion_wc_pos = gr.Image()
                aspect_opinion_hist_pos = gr.Image()
            gr.Markdown("### 3.3 负向属性+观点的内容分析")
            with gr.Row():
                aspect_opinion_wc_neg = gr.Image()
                aspect_opinion_hist_neg = gr.Image()
        gr.Markdown("## 4. 属性 + 极性分析\n 挖掘客户对于产品/服务针对属性的情感极性，帮助商家直观地查看客户对于产品/服务的某些属性的印象。")
        with gr.Row(equal_height=True):
            aspect_sentiment_wc = gr.Image()
            aspect_sentiment_hist = gr.Image()

    upload_file.change(process_upload, inputs=[upload_file], outputs=[upload_file, file_path, msg_box])
    file_btn.click(
        process_file,
        inputs=[file_path],
        outputs=[
            show,
            aspect_wc,
            aspect_hist,
            opinion_wc,
            opinion_hist,
            aspect_opinion_wc,
            aspect_opinion_hist,
            aspect_opinion_wc_pos,
            aspect_opinion_hist_pos,
            aspect_opinion_wc_neg,
            aspect_opinion_hist_neg,
            aspect_sentiment_wc,
            aspect_sentiment_hist,
            msg_box,
        ],
    )
    reset_btn.click(reset_click, inputs=None, outputs=[upload_file, file_path, msg_box, show])


if __name__ == "__main__":
    # To create a public link, set `share=True` in `launch()`.
    demo.launch(enable_queue=False, share=True)
