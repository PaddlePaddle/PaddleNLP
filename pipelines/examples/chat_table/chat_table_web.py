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
from collections import defaultdict

import gradio as gr
from utils import chat_table

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", default="", type=str, help="The API Key.")
parser.add_argument("--secret_key", default="", type=str, help="The secret key.")
args = parser.parse_args()
history_index = defaultdict(list)


def reset_state():
    global history_index
    history_index = defaultdict(list)
    return "", []


def predict(query, history=[], api_key=args.api_key, secret_key=args.secret_key, index=None):
    if index is None:
        index = "document"
    try:
        message = chat_table(query, history_index[index], api_key, secret_key, index)
        history.append(["user: {}".format(query), "assistant: {}".format(message["result"])])
    except:
        history_index[index] = []
        message = chat_table(query, history_index[index], api_key, secret_key, index)
        history.append(["user: {}".format(query), "assistant: {}".format(message["result"])])
    return "", history, history


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">🤖ChatTable</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
                state = gr.State([])
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
    submitBtn.click(predict, [user_input, state], [user_input, chatbot, state])
    emptyBtn.click(reset_state, outputs=[chatbot, state], show_progress=True)
demo.queue().launch(server_name="0.0.0.0", server_port=8084, share=False)
