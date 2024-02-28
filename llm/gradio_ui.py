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


def create_src_slider(value, maximum):
    return gr.Slider(
        minimum=1,
        maximum=maximum,
        value=value,
        step=1,
        label="Max Src Length",
        info="最大输入长度。",
    )


def create_max_slider(value, maximum):
    return gr.Slider(
        minimum=1,
        maximum=maximum,
        value=value,
        step=1,
        label="Max Decoding Length",
        info="生成结果的最大长度。",
    )


def launch(args, default_params: dict = {}):
    """Launch characters dialogue demo."""

    def rollback(state):
        """Rollback context."""
        context = state.setdefault("context", [])
        utterance = context[-2]["utterance"]
        context = context[:-2]
        state["context"] = context
        shown_context = get_shown_context(context)
        return utterance, shown_context, context, state

    def regen(state, top_k, top_p, temperature, repetition_penalty, max_length, src_length):
        """Regenerate response."""
        context = state.setdefault("context", [])
        if len(context) < 2:
            gr.Warning("don't have chat history")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        context.pop()
        user_turn = context.pop()
        context.append({"role": "user", "utterance": user_turn["utterance"]})
        context.append({"role": "bot", "utterance": ""})
        shown_context = get_shown_context(context)
        return user_turn["utterance"], shown_context, context, state

    def begin(utterance, state):
        """Model inference."""
        utterance = utterance.strip().replace("<br>", "\n")
        context = state.setdefault("context", [])

        if not utterance:
            gr.Warning("invalid inputs")
            # gr.Warning("请输入有效问题")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        context.append({"role": "user", "utterance": utterance})
        context.append({"role": "bot", "utterance": ""})

        shown_context = get_shown_context(context)
        return utterance, shown_context, context, state

    def infer(utterance, state, top_k, top_p, temperature, repetition_penalty, max_length, src_length):
        """Model inference."""
        utterance = utterance.strip().replace("<br>", "\n")
        context = state.setdefault("context", [])

        if not utterance:
            gr.Warning("invalid inputs")
            # gr.Warning("请输入有效问题")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        data = {
            "context": utterance,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_length": max_length,
            "src_length": src_length,
            "min_length": 1,
        }
        if len(context) > 2:
            data["history"] = json.dumps(context[:-2])

        res = requests.post(f"http://0.0.0.0:{args.flask_port}/api/chat", json=data, stream=True)
        for index, line in enumerate(res.iter_lines()):
            result = json.loads(line)
            if result["error_code"] != 0:
                gr.Warning(result["error_msg"])
                shown_context = get_shown_context(context)
                return None, shown_context, context, state

            bot_response = result["result"]["response"]

            # replace \n with br: https://github.com/gradio-app/gradio/issues/4344
            bot_response["utterance"] = bot_response["utterance"].replace("\n", "<br>")

            if bot_response["utterance"].endswith("[END]"):
                bot_response["utterance"] = bot_response["utterance"][:-5]

            # the first character of gradio can not be "<br>" or "<br/>"
            if bot_response["utterance"] in ["<br>", "<br/>"] and index == 0:
                continue

            context[-1]["utterance"] += bot_response["utterance"]
            shown_context = get_shown_context(context)

            yield None, shown_context, context, state

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

    with gr.Blocks(title="LLM", theme=gr.themes.Soft()) as block:
        gr.Markdown(f"# {args.title} <font style='color: red !important' size=2>{args.sub_title}</font>")
        with gr.Row():
            with gr.Column(scale=1):
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Top-k",
                    info="该参数越大，模型生成结果更加随机，反之生成结果更加确定。",
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=default_params.get("top_p", 0.7),
                    step=0.05,
                    label="Top-p",
                    info="该参数越大，模型生成结果更加随机，反之生成结果更加确定。",
                )
                temperature = gr.Slider(
                    minimum=0.05,
                    maximum=1.5,
                    value=default_params.get("temperature", 0.95),
                    step=0.05,
                    label="Temperature",
                    info="该参数越小，模型生成结果更加随机，反之生成结果更加确定。",
                )
                repetition_penalty = gr.Slider(
                    minimum=0.1,
                    maximum=10,
                    value=default_params.get("repetition_penalty", 1.2),
                    step=0.05,
                    label="Repetition Penalty",
                    info="该参数越大，生成结果重复的概率越低。设置 1 则不开启。",
                )
                default_src_length = default_params["src_length"]
                total_length = default_params["src_length"] + default_params["max_length"]
                src_length = create_src_slider(default_src_length, total_length)
                max_length = create_max_slider(min(total_length - default_src_length, 50), total_length)

                def src_length_change_event(src_length_value, max_length_value):
                    return create_max_slider(
                        min(total_length - src_length_value, max_length_value),
                        total_length - src_length_value,
                    )

                def max_length_change_event(src_length_value, max_length_value):
                    return create_src_slider(
                        min(total_length - max_length_value, src_length_value),
                        total_length - max_length_value,
                    )

                src_length.change(src_length_change_event, inputs=[src_length, max_length], outputs=max_length)
                max_length.change(max_length_change_event, inputs=[src_length, max_length], outputs=src_length)

            with gr.Column(scale=4):
                state = gr.State({})
                context_chatbot = gr.Chatbot(label="Context")
                utt_text = gr.Textbox(placeholder="请输入...", label="Utterance")
                with gr.Row():
                    clear_btn = gr.Button("清空")
                    rollback_btn = gr.Button("撤回")
                    regen_btn = gr.Button("重新生成")
                    send_btn = gr.Button("发送")
                with gr.Row():
                    raw_context_json = gr.JSON(label="Raw Context")

            utt_text.submit(
                begin,
                inputs=[utt_text, state],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                queue=False,
                api_name="chat",
            ).then(
                infer,
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_length, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
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
                inputs=[state, top_k, top_p, temperature, repetition_penalty, max_length, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                queue=False,
                api_name="chat",
            ).then(
                infer,
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_length, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
            )

            send_btn.click(
                begin,
                inputs=[utt_text, state],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                queue=False,
                api_name="chat",
            ).then(
                infer,
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_length, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
            )

    block.queue().launch(server_name="0.0.0.0", server_port=args.port, debug=True)


def main(args, default_params: dict = {}):
    launch(args, default_params)


if __name__ == "__main__":
    args = setup_args()
    main(args)
