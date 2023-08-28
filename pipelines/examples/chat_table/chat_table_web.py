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

import argparse

import gradio as gr
from chat_table import parsing_QA

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default="", type=str, help="The API Key.")
parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
args = parser.parse_args()


def reset_state():
    return "", []


def predict(query, history=[]):
    result = parsing_QA(args.api_key, args.secret_key, query)
    history.append(["user: {}".format(query), "assistant: {}".format(result)])
    return "", history, history


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">🤖ChatTable</h1>""")
        with gr.Accordion("输出区", open=True, elem_id="input-panel") as area_input_primary:
            chatbot = gr.Chatbot(scale=30, height=600)
        with gr.Accordion("输入区", open=True, elem_id="output-panel") as area_output_primary:
            with gr.Row():
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5, scale=10).style(
                    container=False
                )
                with gr.Column(scale=1):
                    submitBtn = gr.Button("🚀 提交", variant="primary", scale=2, min_width=0)
                    state = gr.State([])
                    emptyBtn = gr.Button("Clear History")
        submitBtn.click(predict, [user_input, state], [user_input, chatbot, state])
        emptyBtn.click(reset_state, outputs=[chatbot, state], show_progress=True)
    demo.queue().launch(server_name="0.0.0.0", server_port=8084, share=False)
