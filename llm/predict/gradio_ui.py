#!/usr/bin/env python
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
import json
import logging
import re

import gradio as gr
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8073)
    parser.add_argument("--api_key", type=str, default=None, help="Your API key")
    parser.add_argument("--model", type=str, default="", help="Model name")
    parser.add_argument("--title", type=str, default="PaddleNLP Chat", help="UI Title")
    parser.add_argument("--sub_title", type=str, default="powered by paddlenlp team.", help="UI Sub Title")
    parser.add_argument("--flask_port", type=int, default=None, help="The port of flask service")
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


def remove_think_tags(text):
    """
    清除文本中 <think> 和 </think> 标签之间的所有字符。

    Args:
        text: 要处理的文本字符串。

    Returns:
        清除 <think> 和 </think> 标签之间内容的文本字符串。
    """
    pattern = re.compile(r"\\<think\\>.*?\\<\\\/think\\>", re.DOTALL)
    # 将匹配到的部分替换为空字符串
    cleaned_text = pattern.sub("", text).strip()
    return cleaned_text


def launch(args, default_params: dict = {}):
    """Launch chat UI with OpenAI API."""

    def rollback(state):
        """Rollback context."""
        context = state.setdefault("context", [])
        # 回退时移除最后一次对话
        if len(context) >= 2:
            content = context[-2]["content"]
            context = context[:-2]
            state["context"] = context
            shown_context = get_shown_context(context)
            return content, shown_context, context, state
        else:
            gr.Warning("没有可撤回的对话历史")
            return None, get_shown_context(context), context, state

    def regen(state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length):
        """Regenerate response."""
        context = state.setdefault("context", [])
        if len(context) < 2:
            gr.Warning("No chat history!")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        # 删除上一次回复，重新生成
        context.pop()
        user_turn = context.pop()
        context.append({"role": "user", "content": user_turn["content"]})
        context.append({"role": "assistant", "content": ""})
        shown_context = get_shown_context(context)
        return user_turn["content"], shown_context, context, state

    def begin(content, state):
        """记录用户输入，并初始化 bot 回复为空。"""
        context = state.setdefault("context", [])

        if not content:
            gr.Warning("Invalid inputs")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        context.append({"role": "user", "content": content})
        context.append({"role": "assistant", "content": ""})
        shown_context = get_shown_context(context)
        return content, shown_context, context, state

    def infer(content, state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length):
        """调用 OpenAI 接口生成回答，并以流式返回部分结果。"""
        context = state.setdefault("context", [])
        if not content:
            gr.Warning("Invalid inputs")
            shown_context = get_shown_context(context)
            return None, shown_context, context, state

        # 构造 OpenAI API 要求的 messages 格式
        messages = []
        for turn in context[:-1]:
            messages.append({"role": turn["role"], "content": remove_think_tags(turn["content"])})

        # 默认模型名称从参数中获取
        model = getattr(args, "model", default_params.get("model", ""))
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
            "src_length": src_length,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
        }
        headers = {
            # "Authorization": "Bearer " + args.api_key,
            "Content-Type": "application/json"
        }
        url = f"http://0.0.0.0:{args.flask_port}/v1/chat/completions"
        try:
            res = requests.post(url, json=payload, headers=headers, stream=True)
        except Exception as e:
            gr.Warning(f"请求异常: {e}")
            shown_context = get_shown_context(context)
            yield None, shown_context, context, state
            return

        # 流式处理返回结果，实时更新最后一个对话记录（即 bot 回复）
        for line in res.iter_lines():
            if line:
                try:
                    decoded_line = line.decode("utf-8").strip()
                    # OpenAI 流返回每行以 "data:" 开头
                    if decoded_line.startswith("data:"):
                        data_str = decoded_line[len("data:") :].strip()
                        if data_str == "[DONE]":
                            logger.info("Conversation round over.")
                            break
                        data_json = json.loads(data_str)

                        # delta 中可能包含部分回复内容
                        delta = data_json["choices"][0]["delta"].get("content", "")
                        if delta:
                            # Reformat <think> tags to show in chatbot
                            delta = delta.replace("<think>", r"\<think\>")
                            delta = delta.replace("</think>", r"\<\/think\>")
                            context[-1]["content"] += delta
                            shown_context = get_shown_context(context)
                            yield None, shown_context, context, state
                    else:
                        logger.error(f"{decoded_line}")
                        gr.Warning(f"{decoded_line}")

                except Exception as e:
                    logger.error(f"解析返回结果异常: {e}")
                    gr.Warning(f"解析返回结果异常: {e}")
                    continue

    def get_shown_context(context):
        """将对话上下文转换为 gr.Chatbot 显示格式，每一对 [用户, 助手]"""
        shown_context = []
        # 每两项组成一对
        for turn_idx in range(0, len(context), 2):
            user_text = context[turn_idx]["content"]
            bot_text = context[turn_idx + 1]["content"] if turn_idx + 1 < len(context) else ""
            shown_context.append([user_text, bot_text])
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
                    info="控制采样token个数。(不建议设置)",
                )
                top_p = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=default_params.get("top_p", 0.7),
                    step=0.05,
                    label="Top-p",
                    info="控制采样范围。",
                )
                temperature = gr.Slider(
                    minimum=0.05,
                    maximum=1.5,
                    value=default_params.get("temperature", 0.95),
                    step=0.05,
                    label="Temperature",
                    info="温度，控制生成随机性。",
                )
                repetition_penalty = gr.Slider(
                    minimum=0.1,
                    maximum=10,
                    value=default_params.get("repetition_penalty", 1.2),
                    step=0.05,
                    label="Repetition Penalty",
                    info="生成结果重复惩罚。(不建议设置)",
                )
                default_src_length = default_params.get("src_length", 128)
                total_length = default_src_length + default_params.get("max_tokens", 50)
                src_length = create_src_slider(default_src_length, total_length)
                max_tokens = create_max_slider(max(total_length - default_src_length, 50), total_length)

                def src_length_change_event(src_length_value, max_tokens_value):
                    return create_max_slider(
                        min(total_length - src_length_value, max_tokens_value),
                        total_length - src_length_value,
                    )

                def max_tokens_change_event(src_length_value, max_tokens_value):
                    return create_src_slider(
                        min(total_length - max_tokens_value, src_length_value),
                        total_length - max_tokens_value,
                    )

                src_length.change(src_length_change_event, inputs=[src_length, max_tokens], outputs=max_tokens)
                max_tokens.change(max_tokens_change_event, inputs=[src_length, max_tokens], outputs=src_length)
            with gr.Column(scale=4):
                state = gr.State({})
                # 这里修改 gr.Chatbot 组件，启用 Markdown 渲染并支持 LaTeX 展示
                context_chatbot = gr.Chatbot(
                    label="Context",
                    render_markdown=True,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True},
                        {"left": "$", "right": "$", "display": True},
                    ],
                )
                utt_text = gr.Textbox(placeholder="请输入...", label="Content")
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
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length],
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
                inputs=[state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
                queue=False,
                api_name="chat",
            ).then(
                infer,
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length],
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
                inputs=[utt_text, state, top_k, top_p, temperature, repetition_penalty, max_tokens, src_length],
                outputs=[utt_text, context_chatbot, raw_context_json, state],
            )

    block.queue().launch(server_name="0.0.0.0", server_port=args.port, debug=True)


def main(args, default_params: dict = {}):
    launch(args, default_params)


if __name__ == "__main__":
    # 可以在 default_params 中设置默认参数，如 src_length, max_tokens, temperature, top_p 等
    default_params = {
        "src_length": 1024,
        "max_tokens": 1024,
        "temperature": 0.95,
        "top_p": 0.7,
    }
    args = setup_args()
    main(args, default_params)
