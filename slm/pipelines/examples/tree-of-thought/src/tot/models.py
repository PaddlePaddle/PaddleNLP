# coding=utf8, ErnestinaQiu
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

import logging
import os

from src.llm import Ernie_llm_list, llamaChatCompletion, llm_config

completion_tokens = prompt_tokens = 0


def completions_with_backoff(**kwargs):
    chatter = kwargs["chatter"]
    return chatter.create(
        messages=kwargs["messages"], temperature=kwargs["temperature"], max_gen_len=kwargs["max_tokens"]
    )


def chatgpt(
    messages, model="llama-2-7b-chat", temperature=0.6, max_tokens=1000, n=1, stop=None, chatter=None, args=None
) -> list:
    global completion_tokens, prompt_tokens
    if chatter is None:
        chatter = llamaChatCompletion(model="llama-2-7b-chat")
        logging.info("Chatter is None. Use llama-2-7b-chat as default.")
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        if model in Ernie_llm_list:
            one_turn_mes = messages[0]  # one_turn_mes is like [{'role': 'user', 'content': "请问你能以《你好，世界》为题，写一首现代诗吗？"}]
            out_content = chatter.create(model=model, messages=one_turn_mes)
            outputs.append(out_content)  # is like ['content']
            # log completion tokens
            completion_tokens += len(out_content)
            prompt_tokens += len(one_turn_mes)
        else:
            res = chatter.create(messages=messages, temperature=temperature)
            outputs.extend([choice["message"]["content"] for choice in res["choices"]])
            # log completion tokens
            completion_tokens += res["usage"]["completion_tokens"]
            prompt_tokens += res["usage"]["prompt_tokens"]

    if args is not None:
        f = open(args.log_fp, "a", encoding="utf8")
        f.write(f"\n [messages]: \n {messages}")
        f.write("\n [outputs]:\n")
        f.write(str(outputs))
        f.close()
    else:
        log_fp = os.path.join(os.getcwd(), "logs", "tot_log.txt")
        os.makedirs(os.path.basename(log_fp), exist_ok=True)
        f.write(f"\n [messages]: \n {messages}")
        f.write("\n [outputs]:\n")
        f.write(str(outputs))
        f.close()
    assert len(outputs) == 1, f"len(outputs) == {len(outputs)}, \n outputs"

    if model in llm_config.keys():
        outputs = outputs[0]
        return outputs
    elif model in Ernie_llm_list:
        return outputs


def gpt(
    prompt, model="llama-2-7b-chat", temperature=0.6, max_tokens=512, n=1, stop=None, args=None, chatter=None
) -> list:
    messages = [[{"role": "user", "content": prompt}]]
    return chatgpt(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=stop,
        args=args,
        chatter=chatter,
    )


def gpt_usage(backend="llama-2-7b-chat"):
    global completion_tokens, prompt_tokens
    cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost,
    }
