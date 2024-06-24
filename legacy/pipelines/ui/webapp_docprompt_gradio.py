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
import base64
import traceback
from io import BytesIO

import cv2
import fitz
import gradio as gr
import numpy as np
import requests
from PIL import Image

fitz_tools = fitz.Tools()

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--serving_name', default="0.0.0.0", help="Serving ip.")
parser.add_argument("--serving_port", default=8891, type=int, help="Serving port.")
args = parser.parse_args()
# yapf: enable


def load_document(path):
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, allow_redirects=True, stream=True)
        b = resp.raw
    else:
        b = open(path, "rb")

    image = Image.open(b)
    images_list = [np.array(image.convert("RGB"))]
    return images_list


def process_path(path):
    error = None
    if path:
        try:
            images_list = load_document(path)
            return (
                path,
                gr.update(visible=True, value=images_list),
                gr.update(visible=True),
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                None,
            )
        except Exception as e:
            traceback.print_exc()
            error = str(e)
    return (
        None,
        gr.update(visible=False, value=None),
        gr.update(visible=False),
        gr.update(visible=False, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=error) if error is not None else None,
        None,
    )


def process_upload(file):
    if file:
        return process_path(file.name)
    else:
        return (
            None,
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            None,
        )


def np2base64(image_np):
    image = cv2.imencode(".jpg", image_np)[1]
    base64_str = str(base64.b64encode(image))[2:-1]
    return base64_str


def get_base64(path):
    if path.startswith("http://") or path.startswith("https://"):
        resp = requests.get(path, allow_redirects=True, stream=True)
        b = resp.raw
    else:
        b = open(path, "rb")

    base64_str = base64.b64encode(b.read()).decode()
    return base64_str


def process_prompt(prompt, document):
    if not prompt:
        prompt = "校验码是多少？"
    if document is None:
        return None, None, None

    url = f"http://{args.serving_name}:{args.serving_port}/query_documents"
    base64_str = get_base64(document)
    r = requests.post(url, json={"meta": {"doc": base64_str, "prompt": [prompt]}})
    response = r.json()
    predictions = response["results"][0]

    pages = [Image.open(BytesIO(base64.b64decode(base64_str)))]

    text_value = predictions[0]["result"][0]["value"]

    return (
        gr.update(visible=True, value=pages),
        gr.update(visible=True, value=predictions),
        gr.update(
            visible=True,
            value=text_value,
        ),
    )


def read_content(file_path: str) -> str:
    """read the content of target file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return content


CSS = """
#prompt input {
    font-size: 16px;
}
#url-textbox {
    padding: 0 !important;
}
#short-upload-box .w-full {
    min-height: 10rem !important;
}
/* I think something like this can be used to re-shape
 * the table
 */
/*
.gr-samples-table tr {
    display: inline;
}
.gr-samples-table .p-2 {
    width: 100px;
}
*/
#select-a-file {
    width: 100%;
}
#file-clear {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
    margin-top: 10px;
}
.gradio-container .gr-button-primary {
    background: linear-gradient(180deg, #CDF9BE 0%, #AFF497 100%);
    border: 1px solid #B0DCCC;
    border-radius: 8px;
    color: #1B8700;
}
.gradio-container.dark button#submit-button {
    background: linear-gradient(180deg, #CDF9BE 0%, #AFF497 100%);
    border: 1px solid #B0DCCC;
    border-radius: 8px;
    color: #1B8700
}
table.gr-samples-table tr td {
    border: none;
    outline: none;
}
table.gr-samples-table tr td:first-of-type {
    width: 0%;
}
div#short-upload-box div.absolute {
    display: none !important;
}
gradio-app > div > div > div > div.w-full > div, .gradio-app > div > div > div > div.w-full > div {
    gap: 0px 2%;
}
gradio-app div div div div.w-full, .gradio-app div div div div.w-full {
    gap: 0px;
}
gradio-app h2, .gradio-app h2 {
    padding-top: 10px;
}
#answer {
    overflow-y: scroll;
    color: white;
    background: #666;
    border-color: #666;
    font-size: 20px;
    font-weight: bold;
}
#answer span {
    color: white;
}
#answer textarea {
    color:white;
    background: #777;
    border-color: #777;
    font-size: 18px;
}
#url-error input {
    color: red;
}
"""

with gr.Blocks(css=CSS) as demo:
    document = gr.Variable()
    example_prompt = gr.Textbox(visible=False)
    example_image = gr.Image(visible=False)
    with gr.Row(equal_height=True):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## 1. Select a file", elem_id="select-a-file")
                img_clear_button = gr.Button("Clear", variant="secondary", elem_id="file-clear", visible=False)
            image = gr.Gallery(visible=False)
            with gr.Row(equal_height=True):
                with gr.Column():
                    with gr.Row():
                        url = gr.Textbox(
                            show_label=False,
                            placeholder="URL",
                            lines=1,
                            max_lines=1,
                            elem_id="url-textbox",
                        )
                        submit = gr.Button("Get")
                    url_error = gr.Textbox(
                        visible=False,
                        elem_id="url-error",
                        max_lines=1,
                        interactive=False,
                        label="Error",
                    )
            gr.Markdown("— or —")
            upload = gr.File(label=None, interactive=True, elem_id="short-upload-box")

        with gr.Column() as col:
            gr.Markdown("## 2. Make a request")
            prompt = gr.Textbox(
                label="Prompt (No restrictions on the setting of prompt. You can type any prompt.)",
                placeholder="e.g. 校验码是多少？",
                lines=1,
                max_lines=1,
            )

            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")
                submit_button = gr.Button("Submit", variant="primary", elem_id="submit-button")
            with gr.Column():
                output_text = gr.Textbox(label="Top Answer", visible=False, elem_id="answer")
                output = gr.JSON(label="Output", visible=False)

    for cb in [img_clear_button, clear_button]:
        cb.click(
            lambda _: (
                gr.update(visible=False, value=None),
                None,
                gr.update(visible=False, value=None),
                gr.update(visible=False, value=None),
                gr.update(visible=False),
                None,
                None,
                None,
                gr.update(visible=False, value=None),
                None,
            ),
            inputs=clear_button,
            outputs=[
                image,
                document,
                output,
                output_text,
                img_clear_button,
                example_image,
                upload,
                url,
                url_error,
                prompt,
            ],
        )

    upload.change(
        fn=process_upload,
        inputs=[upload],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )
    submit.click(
        fn=process_path,
        inputs=[url],
        outputs=[document, image, img_clear_button, output, output_text, url_error],
    )

    prompt.submit(
        fn=process_prompt,
        inputs=[prompt, document],
        outputs=[image, output, output_text],
    )

    submit_button.click(
        fn=process_prompt,
        inputs=[prompt, document],
        outputs=[image, output, output_text],
    )

if __name__ == "__main__":
    # To create a public link, set `share=True` in `launch()`.
    demo.launch(enable_queue=False, share=True)
