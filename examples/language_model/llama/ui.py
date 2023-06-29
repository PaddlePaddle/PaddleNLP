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

from __future__ import annotations

import argparse
import copy
import json

import gradio as gr
import requests


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8073)
    args = parser.parse_args()
    return args


def launch(args):
    """Launch characters dialogue demo."""

    def rollback(state):
        """Rollback context."""
        context = state.setdefault("context", [])
        utterance = context[-2]["utterance"]
        context = context[:-2]
        state["context"] = context
        shown_context = get_shown_context(context)
        return utterance, shown_context, context, state

    def regen(state, version, top_p, temperature):
        """Regenerate response."""
        context = state.setdefault("context", [])
        context.pop()
        user_turn = context.pop()
        return infer(user_turn["utterance"], state, version, top_p, temperature)

    def infer(utterance, state, top_p, temperature, max_length):
        """Model inference."""
        utterance = utterance.strip().replace("<br>", "\n")
        context = state.setdefault("context", [])
        context.append({"role": "user", "utterance": utterance})
        data = {
            "context": utterance,
            "top_p": top_p,
            "temperature": temperature,
            "max_length": max_length,
            "min_length": 1,
        }
        result = requests.post(f"http://0.0.0.0:{args.flask_port}/api/chat", json=data).json()
        bot_response = result["result"]["response"]
        context.append(bot_response)
        shown_context = get_shown_context(context)
        return None, shown_context, context, state

    def clean_context(context):
        """Clean context for EB input."""
        cleaned_context = copy.deepcopy(context)
        for turn in cleaned_context:
            if turn["role"] == "bot":
                bot_resp = turn["utterance"]
                if bot_resp.startswith("<img src") or bot_resp.startswith("<audio controls>"):
                    bot_resp = "\n".join(bot_resp.split("\n")[1:])
                turn["utterance"] = bot_resp
        return cleaned_context

    def extract_eda(eb_debug_info):
        """Extract EDA result from EB dispatch info."""
        eda_res = None
        for item in eb_debug_info:
            if item["sys"] == "EDA":
                eda_output = json.loads(item["output"])
                eda_res = eda_output["result"]
                break
        return eda_res

    def extract_eb_input(eb_debug_info, convert_for_ar=True):
        """Extract EB raw input from EB dispatch info."""
        eb_raw_input = None
        for item in eb_debug_info:
            if item["sys"] == "EB":
                eb_output = json.loads(item["output"])
                eb_raw_input = eb_output["text_after_process"]
                if convert_for_ar:
                    eb_raw_input = eb_raw_input.replace("[CLS]", "<cls>").replace("[SEP]", "<sep>")
                break
        return eb_raw_input

    def get_shown_context(context):
        """Get gradio chatbot."""
        shown_context = []
        for turn_idx in range(0, len(context), 2):
            shown_context.append([context[turn_idx]["utterance"], context[turn_idx + 1]["utterance"]])
        return shown_context

    with gr.Blocks(title="LLaMA", theme=gr.themes.Soft()) as block:
        gr.Markdown("# LLaMA")
        with gr.Row():
            with gr.Column(scale=1):
                top_p = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.05, label="Top-p")
                temperature = gr.Slider(minimum=0.05, maximum=1.5, value=0.95, step=0.05, label="Temperature")
                max_length = gr.Slider(minimum=1, maximum=1024, value=10, step=1, label="Max Length")
            with gr.Column(scale=4):
                state = gr.State({})
                context_chatbot = gr.Chatbot(label="context")
                utt_text = gr.Textbox(placeholder="请输入...", label="utterance")
                with gr.Row():
                    clear_btn = gr.Button("清空")
                    rollback_btn = gr.Button("撤回")
                    regen_btn = gr.Button("重新生成")
                    send_btn = gr.Button("发送")
                with gr.Row():
                    raw_context_json = gr.JSON(label="raw context")

            utt_text.submit(
                infer,
                inputs=[utt_text, state, top_p, temperature, max_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                api_name="chat",
            )
            clear_btn.click(
                lambda _: (None, None, None, {}),
                inputs=clear_btn,
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                api_name="clear",
                show_progress=False,
            )
            rollback_btn.click(
                rollback,
                inputs=[state],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                show_progress=False,
            )
            regen_btn.click(
                regen,
                inputs=[state, top_p, temperature, max_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
            )
            send_btn.click(
                infer,
                inputs=[utt_text, state, top_p, temperature, max_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
            )

    block.launch(server_name="0.0.0.0", server_port=args.port, debug=True)


def main(args):
    launch(args)


if __name__ == "__main__":
    args = setup_args()
    main(args)
